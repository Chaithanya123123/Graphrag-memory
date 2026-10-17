"""FalkorDB context layer with in-memory fallback (works before docker is ready)."""
import os, time, re, uuid

HOST = os.getenv("FALKOR_HOST", "localhost")
PORT = int(os.getenv("FALKOR_PORT", "6379"))

_db = None
_mode = "memory"
try:
    from falkordb import FalkorDB
    _db = FalkorDB(host=HOST, port=PORT)
    _db.list_graphs()
    _mode = "falkordb"
    print(f"[graph] FalkorDB connected at {HOST}:{PORT}")
except Exception as e:
    print(f"[graph] FalkorDB unavailable ({e}) -> DEMO memory mode")

# ---- in-memory fallback store: {graph_name: {"nodes":[], "edges":[], "episodes":[...]}} ----
_MEM = {}

# ---- single seed definition: BOTH backends build from this. No drift. ----
# One shared SEED structure so init_graph (falkordb) and the memory fallback can
# never diverge again — every recent FalkorDB bug was a mode mismatch.
SEED = {
    "papers": [
        {"id": "p1", "title": "Graph RAG Survey", "year": 2024, "citations": 40, "venue": "ACL"},
        {"id": "p2", "title": "FalkorDB Memory for Agents", "year": 2025, "citations": 12, "venue": ""},
        {"id": "p3", "title": "Vector-Only Retrieval Limits", "year": 2023, "citations": 80, "venue": "NeurIPS"},
    ],
    "methods": ["GraphRAG", "VectorRAG"],
    "datasets": ["HotpotQA"],
    "claims": [
        {"id": "c1", "text": "Graph retrieval beats vector-only on multi-hop QA"},
        {"id": "c2", "text": "Vector-only is sufficient for QA"},
    ],
    "people": [{"name": "Chaithanya", "role": "Researcher", "key": "seed:a"}],
    "projects": [{"name": "TraceMind", "status": "active"}],
    # (from_label, from_prop, from_val, REL, to_label, to_prop, to_val, edge_props)
    # labels/rel/prop are code constants, never request data — safe to inline in Cypher.
    "edges": [
        ("Paper", "id", "p2", "CITES", "Paper", "id", "p1", {"source": "seed"}),
        ("Paper", "id", "p1", "CITES", "Paper", "id", "p3", {"source": "seed"}),
        ("Person", "name", "Chaithanya", "WROTE", "Paper", "id", "p1", {"source": "seed"}),
        ("Person", "name", "Chaithanya", "WROTE", "Paper", "id", "p2", {"source": "seed"}),
        ("Paper", "id", "p1", "USES_METHOD", "Method", "name", "GraphRAG", {}),
        ("Paper", "id", "p2", "USES_METHOD", "Method", "name", "GraphRAG", {}),
        ("Paper", "id", "p3", "USES_METHOD", "Method", "name", "VectorRAG", {}),
        ("Paper", "id", "p1", "ABOUT", "Topic", "name", "GraphRAG", {}),
        ("Paper", "id", "p2", "ABOUT", "Topic", "name", "GraphRAG", {}),
        ("Paper", "id", "p3", "ABOUT", "Topic", "name", "VectorRAG", {}),
        ("Paper", "id", "p1", "CLAIMS", "Claim", "id", "c1", {"source": "seed"}),
        ("Paper", "id", "p2", "CLAIMS", "Claim", "id", "c1", {"source": "seed"}),
        ("Paper", "id", "p3", "CLAIMS", "Claim", "id", "c2", {"source": "seed"}),
        ("Claim", "id", "c1", "CONTRADICTS", "Claim", "id", "c2", {"evidence": "seed demo pair"}),
        ("Person", "name", "Chaithanya", "WORKS_ON", "Project", "name", "TraceMind", {}),
        ("Paper", "id", "p2", "EVALUATED_ON", "Dataset", "name", "HotpotQA", {}),
    ],
}
SEED_TOPICS = sorted({e[6] for e in SEED["edges"] if e[3] == "ABOUT"})
_SEED_CITES = [(e[2], e[6]) for e in SEED["edges"] if e[3] == "CITES"]

def mode():
    return _mode

def _mem(gname):
    if gname not in _MEM:
        pkey = {p["name"]: p["key"] for p in SEED["people"]}
        m = {"episodes": [], "papers": [dict(p) for p in SEED["papers"]],
             "seeded": False, "claims": [],
             "authored": [], "methods": {}, "paper_topic": {}}
        # same edges, same structure the falkordb branch writes
        for e in SEED["edges"]:
            fl, fp, fv, rel, tl, tp, tv, _props = e
            if rel == "WROTE":
                m["authored"].append({"key": pkey.get(fv, "seed"), "name": fv, "paper": tv})
            elif rel == "USES_METHOD":
                m["methods"].setdefault(fv, []).append(tv)
            elif rel == "ABOUT":
                m["paper_topic"][fv] = tv
        _MEM[gname] = m
    return _MEM[gname]

def sanitize_user_id(user_id: str) -> str:
    if not re.fullmatch(r"[a-z0-9_]{1,32}", user_id):
        raise ValueError("bad user id: must be lowercase alphanumeric + underscore, 1-32 chars")
    return user_id

def graph_name(user_id: str) -> str:
    return f"user_{sanitize_user_id(user_id)}"

def user_graph(user_id: str):
    if _mode == "falkordb":
        return _db.select_graph(graph_name(user_id))
    return None

def _seed_node_cypher(label):
    """MERGE statement per seed label — keyed on id or name, MERGE makes
    init_graph idempotent: calling it twice changes no node/edge counts."""
    return {
        "Paper": "MERGE (n:Paper {id:$v}) SET n.title=$title, n.year=$year, "
                 "n.citations=$cites, n.venue=$venue, n.source='seed'",
        "Method": "MERGE (n:Method {name:$v})",
        "Dataset": "MERGE (n:Dataset {name:$v})",
        "Topic": "MERGE (n:Topic {name:$v})",
        "Claim": "MERGE (n:Claim {id:$v}) SET n.text=$text",
        "Person": "MERGE (n:Person {name:$v}) SET n.role=$role, n.key=$key",
        "Project": "MERGE (n:Project {name:$v}) SET n.status=$status",
    }[label]


def _seed_node_params(node_def):
    """Accept either a bare string (name-keyed labels) or a dict."""
    if isinstance(node_def, str):
        return {"v": node_def, "title": "", "year": 0, "cites": 0, "venue": "",
                "text": "", "role": "", "key": "", "status": ""}
    return {"v": node_def.get("id") or node_def.get("name"),
            "title": node_def.get("title", ""), "year": node_def.get("year", 0),
            "cites": node_def.get("citations", 0), "venue": node_def.get("venue", ""),
            "text": node_def.get("text", ""), "role": node_def.get("role", ""),
            "key": node_def.get("key", ""), "status": node_def.get("status", "")}


def init_graph(user_id: str):
    if _mode == "falkordb":
        g = user_graph(user_id)
        try:
            node_defs = ([("Paper", p) for p in SEED["papers"]]
                         + [("Method", x) for x in SEED["methods"]]
                         + [("Dataset", x) for x in SEED["datasets"]]
                         + [("Topic", x) for x in SEED_TOPICS]
                         + [("Claim", c) for c in SEED["claims"]]
                         + [("Person", p) for p in SEED["people"]]
                         + [("Project", p) for p in SEED["projects"]])
            for label, nd in node_defs:
                g.query(_seed_node_cypher(label), params=_seed_node_params(nd))
            for fl, fp, fv, rel, tl, tp, tv, props in SEED["edges"]:
                q = (f"MATCH (a:{fl} {{{fp}:$fv}}), (b:{tl} {{{tp}:$tv}}) "
                     f"MERGE (a)-[r:{rel}]->(b)")
                if props:
                    q += " SET " + ", ".join(f"r.{k}=$p_{k}" for k in props)
                g.query(q, params={"fv": fv, "tv": tv,
                                   **{f"p_{k}": v for k, v in props.items()}})
        except Exception as e:
            print("seed warn:", e)
        return {"ok": True, "mode": _mode}
    _mem(graph_name(user_id))["seeded"] = True
    return {"ok": True, "mode": _mode}

def remember(user_id, fact):
    if _mode == "falkordb":
        try:
            g = user_graph(user_id)
            g.query("MERGE (u:User {id:$uid}) MERGE (e:Episode {text:$f, at:timestamp()}) MERGE (u)-[:REMEMBERS]->(e)",
                    params={"uid": sanitize_user_id(user_id), "f": fact})
        except Exception as e:
            print("remember warn:", e)
        return
    _mem(graph_name(user_id))["episodes"].append({"text": fact, "at": time.time()})

def recall(user_id, limit=10):
    if _mode == "falkordb":
        try:
            g = user_graph(user_id)
            r = g.query("MATCH (u:User {id:$uid})-[:REMEMBERS]->(e:Episode) RETURN e.text AS m ORDER BY e.at DESC LIMIT 10",
                        params={"uid": sanitize_user_id(user_id)})
            return [row[0] for row in r.result_set]
        except Exception:
            return []
    eps = _mem(graph_name(user_id))["episodes"]
    return [e["text"] for e in reversed(eps[-limit:])]

def _snap_nid(n):
    # key nodes by their `id` property (doi:/s2:/oa:/c...) when present:
    # PageRank ranking, highlight pinning and memory-mode all use property ids,
    # so the internal integer id would never match.
    props = dict(n.properties or {})
    return str(props.get("id") or n.id)


def _rows_to_viz(rows):
    """[(nodeA, rel, nodeB, source, retrieved_at, url, evidence)] -> viz dicts."""
    nodes, edges, seen = [], [], set()
    for row in rows:
        a, rel, b = row[0], row[1], row[2]
        for n in (a, b):
            nid = _snap_nid(n)
            if nid not in seen:
                seen.add(nid)
                props = dict(n.properties or {})
                label = (props.get("title") or props.get("name")
                         or (props.get("text") or "")[:28]
                         or (n.labels[0] if n.labels else nid))
                nodes.append({
                    "id": nid, "label": label,
                    "type": (n.labels[0] if n.labels else "Node"),
                    "updated_at": props.get("updated_at"),
                    "citations": props.get("citations"),
                    "url": props.get("url", ""), "source": props.get("source", "")})
        edges.append({"from": _snap_nid(a), "to": _snap_nid(b), "label": rel,
                      "source": row[3] or "", "retrieved_at": row[4],
                      "url": row[5] or "", "evidence": row[6] or ""})
    return nodes, edges


def graph_snapshot(user_id: str, top_n: int = 40, highlight: list = None):
    """Nodes+edges for Cytoscape viz.

    A single ingest pulls in hundreds of hydrated reference stubs; showing all of
    them buries the interesting papers. Keep the top_n Papers by influence plus
    everything on the highlighted path.

    FalkorDB mode is SELECTION-DRIVEN: it asks the DB for the edges incident to
    exactly those papers. Trimming an arbitrary first-N-edge window instead was
    intersecting two different node sets (window papers vs cited-ranked papers)
    and dropped every edge — the panel showed bare nodes and traceability died.
    """
    if _mode == "falkordb":
        g = user_graph(user_id)
        try:
            total_n = g.query("MATCH (n) RETURN count(n)").result_set[0][0]
            total_e = g.query("MATCH ()-[r]->() RETURN count(r)").result_set[0][0]
            ranked_ids = [row[0] for row in g.query(
                "MATCH (p:Paper) RETURN p.id ORDER BY coalesce(p.citations,0) DESC LIMIT $k",
                params={"k": top_n}).result_set if row[0]]
        except Exception as e:
            print("snapshot rank warn:", e)
            return {"nodes": [], "edges": [], "total_nodes": 0, "total_edges": 0, "hidden": 0}
        ids = list({*ranked_ids, *(highlight or [])})
        try:
            r = g.query(
                "MATCH (a)-[rel]->(b) WHERE a.id IN $ids OR b.id IN $ids "
                "RETURN a, type(rel), b, rel.source, rel.retrieved_at, rel.url, rel.evidence "
                "LIMIT 300", params={"ids": ids})
            nodes, edges = _rows_to_viz(r.result_set)
            return {"nodes": nodes, "edges": edges,
                    "total_nodes": total_n, "total_edges": total_e,
                    "hidden": max(0, total_n - len(nodes))}
        except Exception as e:
            print("snapshot warn:", e)
            return {"nodes": [], "edges": [], "total_nodes": 0, "total_edges": 0, "hidden": 0}
    # memory mode: the whole store is one consistent window, filter locally
    full = _snapshot_all(user_id)
    if not full["nodes"] or top_n <= 0:
        return full
    keep = {n["id"] for n in full["nodes"] if n.get("type") != "Paper"}
    keep |= set(highlight or [])
    ranked_ids = [r["id"] for r in _pagerank_py(user_id, 0.85, 20, top_n)]
    for pid in ranked_ids:
        if len(keep) >= top_n + 10:
            break
        keep.add(pid)
    nodes = [n for n in full["nodes"] if n["id"] in keep]
    edges = [e for e in full["edges"] if e["from"] in keep and e["to"] in keep]
    return {"nodes": nodes, "edges": edges,
            "total_nodes": len(full["nodes"]), "total_edges": len(full["edges"]),
            "hidden": len(full["nodes"]) - len(nodes)}


def _snapshot_all(user_id: str):
    if _mode == "falkordb":
        try:
            g = user_graph(user_id)
            r = g.query("MATCH (a)-[rel]->(b) RETURN a, type(rel), b, "
                        "rel.source, rel.retrieved_at, rel.url, rel.evidence LIMIT 60")
            nodes, edges = _rows_to_viz(r.result_set)
            return {"nodes": nodes, "edges": edges}
        except Exception as e:
            print("snapshot warn:", e)
    # memory fallback snapshot — generated from the SAME SEED the falkordb
    # branch MERGEs, so the two backends cannot drift into different stories
    def _mkey(label, val):
        return val if label in ("Paper", "Claim") else f"{label.lower()}:{val}"

    base = {"nodes": [], "edges": []}
    def _add_node(nid, label, ntype, cites=None, source="seed"):
        if not any(n["id"] == nid for n in base["nodes"]):
            base["nodes"].append({"id": nid, "label": label, "type": ntype,
                                  "updated_at": None, "citations": cites,
                                  "url": "", "source": source})
    for p in SEED["papers"]:
        _add_node(p["id"], p["title"], "Paper", p.get("citations", 0))
    for c in SEED["claims"]:
        _add_node(c["id"], c["text"][:32], "Claim")
    for e in SEED["edges"]:
        fl, fp, fv, rel, tl, tp, tv, props = e
        _add_node(_mkey(fl, fv), fv if fl not in ("Paper", "Claim") else "", fl)
        _add_node(_mkey(tl, tv), tv if tl not in ("Paper", "Claim") else "", tl)
        base["edges"].append({"from": _mkey(fl, fv), "to": _mkey(tl, tv), "label": rel,
                              "source": props.get("source", "seed"), "retrieved_at": None,
                              "url": "", "evidence": props.get("evidence", "")})
    for n in base["nodes"]:
        if n["type"] == "Paper" and not n["label"]:
            n["label"] = next(p["title"] for p in SEED["papers"] if p["id"] == n["id"])[:40]
    # merge live ingest: papers, real CITES edges, claims, contradictions, supersedes
    try:
        m = _mem(graph_name(user_id))
        have = {n["id"] for n in base["nodes"]}
        for p in m.get("papers", []):
            if p["id"] in have:
                continue
            have.add(p["id"])
            base["nodes"].append({"id": p["id"], "label": (p.get("title") or p["id"])[:40],
                                  "type": "Paper", "updated_at": p.get("updated_at"),
                                  "citations": p.get("citations", 0), "url": p.get("url", ""),
                                  "source": p.get("source", "live")})
        for c in m.get("claims", [])[-30:]:
            if c["id"] in have:
                continue
            have.add(c["id"])
            base["nodes"].append({"id": c["id"], "label": c["text"][:32], "type": "Claim",
                                  "updated_at": c.get("updated_at"), "citations": None,
                                  "url": "", "source": c.get("source", "")})
            if c.get("paper_id"):
                base["edges"].append({"from": c["paper_id"], "to": c["id"], "label": "CLAIMS",
                                      "source": c.get("source", ""), "retrieved_at": c.get("updated_at"),
                                      "url": "", "evidence": ""})
        for e in m.get("cites", [])[-80:]:
            base["edges"].append({"from": e["from"], "to": e["to"], "label": "CITES",
                                  "source": e.get("source", ""), "retrieved_at": None,
                                  "url": "", "evidence": ""})
        for l in m.get("claim_links", [])[-20:]:
            base["edges"].append({"from": l["a_id"], "to": l["b_id"],
                                  "label": "CONTRADICTS" if l["relation"] == "contradicts" else "SUPPORTS",
                                  "source": "llm", "retrieved_at": None, "url": "",
                                  "evidence": l.get("evidence", "")})
        for s in m.get("superseded", [])[-10:]:
            base["edges"].append({"from": s["from"], "to": s["to"], "label": "SUPERSEDED_BY",
                                  "source": "reingest", "retrieved_at": None, "url": "", "evidence": ""})
    except Exception as e:
        print("mem snapshot warn:", e)
    return base


# ---- real claims store + query (used by sources.store_papers + agent) ----

def _claim_fingerprint(text: str) -> str:
    """Stable key for 'the same claim said differently' so re-ingest can supersede."""
    import hashlib
    norm = "".join(ch.lower() for ch in (text or "") if ch.isalnum())
    return hashlib.sha1(norm[:180].encode("utf-8")).hexdigest()[:16]


def store_claim_nodes(user_id: str, paper_id: str, paper_title: str, topic: str,
                      claims: list, source: str):
    """Persist (Paper)-[:CLAIMS]->(Claim)-[:ABOUT]->(Topic) with provenance.

    Re-ingesting the same paper refreshes updated_at. A newer claim with the same
    fingerprint supersedes the older node instead of overwriting it, so the
    history of what was believed stays queryable."""
    import time as _time
    if _mode == "falkordb":
        g = user_graph(user_id)
        try:
            for c in claims:
                fp = c.get("fingerprint") or _claim_fingerprint(c["text"])
                now = int(_time.time())
                g.query(
                    "MERGE (c:Claim {id:$cid}) "
                    "ON CREATE SET c.created_at=$now "
                    "SET c.text=$text, c.source=$src, c.retrieved_at=$now, "
                    "c.paper_id=$pid, c.fingerprint=$fp, c.updated_at=$now",
                    params={"cid": c["id"], "text": c["text"][:250], "src": source[:20],
                            "pid": paper_id, "fp": fp, "now": now})
                g.query(
                    "MATCH (p:Paper {id:$pid}), (c:Claim {id:$cid}) "
                    "MERGE (p)-[r:CLAIMS]->(c) "
                    "SET r.source=$src, r.retrieved_at=$now, r.updated_at=$now",
                    params={"pid": paper_id, "cid": c["id"], "src": source[:20], "now": now})
                if topic:
                    g.query("MERGE (t:Topic {name:$t}) WITH t MATCH (c:Claim {id:$cid}) "
                            "MERGE (c)-[:ABOUT]->(t)",
                            params={"t": topic[:80], "cid": c["id"]})
                # knowledge currency: newer version of the same claim supersedes the old
                g.query(
                    "MATCH (old:Claim {fingerprint:$fp}), (new:Claim {id:$cid}) "
                    "WHERE old.id <> new.id AND coalesce(old.updated_at,0) < coalesce(new.updated_at,0) "
                    "MERGE (old)-[:SUPERSEDED_BY]->(new)",
                    params={"fp": fp, "cid": c["id"]})
        except Exception as e:
            print("store_claim warn:", e)
    else:
        m = _mem(graph_name(user_id))
        m.setdefault("claims", [])
        m.setdefault("claim_links", [])
        have = {x["id"] for x in m["claims"]}
        for c in claims:
            fp = c.get("fingerprint") or _claim_fingerprint(c["text"])
            for old in m["claims"]:
                if old.get("fingerprint") == fp and old["id"] != c["id"] \
                        and old.get("updated_at", 0) <= _time.time():
                    m.setdefault("superseded", []).append(
                        {"from": old["id"], "to": c["id"]})
            rec = {"id": c["id"], "text": c["text"], "paper_id": paper_id,
                   "paper": paper_title, "topic": topic, "source": source,
                   "fingerprint": fp, "updated_at": _time.time()}
            if c["id"] in have:
                m["claims"] = [rec if x["id"] == c["id"] else x for x in m["claims"]]
            else:
                m["claims"].append(rec)
                have.add(c["id"])


def store_claim_links(user_id: str, links: list):
    """Persist (Claim)-[:CONTRADICTS|:SUPPORTS {evidence}]->(Claim). Rel whitelisted."""
    if not links:
        return
    if _mode == "falkordb":
        g = user_graph(user_id)
        for l in links:
            rel = "CONTRADICTS" if l["relation"] == "contradicts" else "SUPPORTS"
            try:
                # rel is whitelisted above, ids/evidence are parameters (no injection)
                g.query(
                    f"MATCH (a:Claim {{id:$a}}), (b:Claim {{id:$b}}) "
                    f"MERGE (a)-[r:{rel}]->(b) SET r.evidence=$ev, r.source=$src, r.retrieved_at=timestamp()",
                    params={"a": l["a_id"], "b": l["b_id"],
                            "ev": l.get("evidence", "")[:250], "src": "llm"})
            except Exception as e:
                print("store_link warn:", e)
    else:
        m = _mem(graph_name(user_id))
        m.setdefault("claim_links", [])
        have = {(x["a_id"], x["b_id"], x["relation"]) for x in m["claim_links"]}
        for l in links:
            k = (l["a_id"], l["b_id"], l["relation"])
            if k not in have:
                m["claim_links"].append(l)
                have.add(k)


def query_contradictions(user_id: str, limit: int = 10):
    """Real contradictions from graph: (Paper)-[:CLAIMS]->(Claim)-[CONTRADICTS]->(Claim)<-[:CLAIMS]-(Paper)."""
    if _mode == "falkordb":
        try:
            g = user_graph(user_id)
            r = g.query(
                "MATCH (p1:Paper)-[:CLAIMS]->(c1:Claim)-[r:CONTRADICTS]->(c2:Claim)"
                "<-[:CLAIMS]-(p2:Paper) "
                "RETURN p1.title AS t1, c1.text AS x1, c2.text AS x2, p2.title AS t2, "
                "r.evidence AS ev LIMIT $lim",
                params={"lim": limit})
            return [{"p1": row[0], "c1": row[1], "c2": row[2], "p2": row[3],
                     "evidence": row[4]} for row in r.result_set]
        except Exception as e:
            print("query_contra warn:", e)
            return []
    m = _mem(graph_name(user_id))
    by_id = {c["id"]: c for c in m.get("claims", [])}
    papers = {}
    try:
        papers = {p["id"]: p.get("title", p["id"]) for p in m.get("papers", [])}
    except Exception:
        pass
    out = []
    for l in m.get("claim_links", [])[-limit:]:
        if l["relation"] != "contradicts":
            continue
        a, b = by_id.get(l["a_id"], {}), by_id.get(l["b_id"], {})
        out.append({"p1": papers.get(a.get("paper_id", ""), a.get("paper", "")),
                    "c1": a.get("text", l["a_id"]), "c2": b.get("text", l["b_id"]),
                    "p2": papers.get(b.get("paper_id", ""), b.get("paper", "")),
                    "evidence": l.get("evidence", "")})
    return out


# ---- Graph Algorithm Tools ----
# Preferred: FalkorDB `algo.*` procedures. Fallback: identical algorithm in Python
# so a demo machine without the procedures still returns a real, ranked answer.

def _citations(user_id: str):
    """[(citing_id, cited_id)] over CITES, from either backend."""
    if _mode == "falkordb":
        try:
            r = user_graph(user_id).query("MATCH (a:Paper)-[:CITES]->(b:Paper) RETURN a.id, b.id")
            return [(row[0], row[1]) for row in r.result_set if row[0] and row[1]]
        except Exception as e:
            print("citations warn:", e)
            return []
    # seed chain first: lineage/PageRank/BFS must see p2->p1->p3 even before ingest
    return _SEED_CITES + [(c["from"], c["to"]) for c in _mem(graph_name(user_id)).get("cites", [])]


def _paper_index(user_id: str) -> dict:
    """{paper_id: {title, year, citations}} from either backend."""
    if _mode == "falkordb":
        try:
            r = user_graph(user_id).query(
                "MATCH (p:Paper) RETURN p.id, p.title, p.year, p.citations LIMIT 2000")
            return {row[0]: {"title": row[1] or row[0], "year": row[2], "citations": row[3] or 0}
                    for row in r.result_set if row[0]}
        except Exception:
            return {}
    return {p["id"]: {"title": p.get("title", p["id"]), "year": p.get("year"),
                      "citations": p.get("citations", 0)}
            for p in _mem(graph_name(user_id)).get("papers", [])}


_PROC_CACHE: dict = {}


def _have(user_id: str, name: str) -> bool:
    """Is this FalkorDB procedure available in the connected version? Cached."""
    if _mode != "falkordb":
        return False
    if name in _PROC_CACHE:
        return _PROC_CACHE[name]
    try:
        r = user_graph(user_id).query(
            "CALL dbms.procedures() YIELD name WHERE name = $n RETURN name LIMIT 1",
            params={"n": name})
        _PROC_CACHE[name] = bool(r.result_set)
    except Exception:
        _PROC_CACHE[name] = False
    return _PROC_CACHE[name]


def algorithm_engine(user_id: str) -> dict:
    """Which implementation actually ran, so the demo never overclaims."""
    return {"pagerank": "falkordb" if _have(user_id, "algo.pagerank") else "python",
            "community": "falkordb" if _have(user_id, "algo.louvain") else "python",
            "shortest_path": "falkordb" if _mode == "falkordb" else "python",
            "components": "python",
            "backend": _mode}


def _pagerank_py(user_id: str, damping: float, max_iter: int, top_k: int):
    """Standard PageRank: dangling mass redistributes uniformly (keeps score mass = 1)."""
    idx = _paper_index(user_id)
    edges = _citations(user_id)
    nodes = [n for n in idx if n in {a for a, _ in edges} | {b for _, b in edges}]
    if not nodes:
        return []
    n = len(nodes)
    pos = {v: i for i, v in enumerate(nodes)}
    out_links = [[] for _ in range(n)]
    in_links = [[] for _ in range(n)]
    for a, b in edges:
        if a in pos and b in pos and a != b:
            out_links[pos[a]].append(pos[b])
            in_links[pos[b]].append(pos[a])
    score = [1.0 / n] * n
    for _ in range(max_iter):
        dangling = sum(score[i] for i in range(n) if not out_links[i])
        base = (1 - damping) / n + damping * dangling / n
        nxt = []
        for i in range(n):
            s = base + damping * sum(score[j] for j in in_links[i]) / max(1, len(in_links[i]))
            nxt.append(s)
        total = sum(nxt) or 1.0
        score = [v / total for v in nxt]
    ranked = sorted(range(n), key=lambda i: -score[i])[:top_k]
    return [{"id": nodes[i], "title": idx[nodes[i]]["title"], "score": score[i]} for i in ranked]


def run_pagerank(user_id: str, max_iter: int = 20, damping: float = 0.85, top_k: int = 20):
    """Most influential papers: PageRank over CITES edges."""
    if _have(user_id, "algo.pagerank"):
        try:
            r = user_graph(user_id).query(
                "CALL algo.pagerank('Paper', 'CITES', {iterations: $iter, dampingFactor: $damp}) "
                "YIELD nodeId, score "
                "MATCH (p:Paper) WHERE id(p) = nodeId "
                "RETURN p.id AS id, p.title AS title, score "
                "ORDER BY score DESC LIMIT $k",
                params={"iter": max_iter, "damp": damping, "k": top_k})
            if r.result_set:
                return [{"id": row[0], "title": row[1], "score": row[2]} for row in r.result_set]
        except Exception as e:
            print("pagerank warn:", e)
    return _pagerank_py(user_id, damping, max_iter, top_k)


def _bfs_py(user_id: str, paper_a: str, paper_b: str, max_depth: int):
    """Undirected BFS over CITES (citations are traversed both ways for lineage)."""
    adj = {}
    for a, b in _citations(user_id):
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    if paper_a not in adj or paper_b not in adj:
        return []
    prev, queue, seen = {paper_a: None}, [paper_a], {paper_a}
    while queue:
        cur = queue.pop(0)
        if cur == paper_b:
            path = [paper_b]
            while prev[path[-1]] is not None:
                path.append(prev[path[-1]])
            return list(reversed(path))
        for nxt in adj.get(cur, ()):
            if nxt in seen:
                continue
            seen.add(nxt)
            prev[nxt] = cur
            queue.append(nxt)
        if len(seen) > 5000:
            break
    return []


def shortest_path_between(user_id: str, paper_a: str, paper_b: str, max_depth: int = 6):
    """Shortest citation path between two papers, with edge provenance for hover."""
    if _mode == "falkordb":  # shortestPath() is a native Cypher function
        try:
            r = user_graph(user_id).query(
                "MATCH (a:Paper {id:$a}), (b:Paper {id:$b}) "
                "MATCH p = shortestPath((a)-[:CITES*..$depth]-(b)) "
                "RETURN [n IN nodes(p) | {id: n.id, title: n.title}] AS path",
                params={"a": paper_a, "b": paper_b, "depth": max_depth})
            if r.result_set:
                return r.result_set[0][0]
            return []
        except Exception as e:
            print("shortest_path warn:", e)
    ids = _bfs_py(user_id, paper_a, paper_b, max_depth)[:max_depth + 1]
    idx = _paper_index(user_id)
    return [{"id": i, "title": idx.get(i, {}).get("title", i)} for i in ids]


def _louvain_py(user_id: str, min_size: int):
    """Label-propagation clustering on the undirected CITES graph (community ids)."""
    adj = {}
    for a, b in _citations(user_id):
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    if not adj:
        return []
    labels = {n: n for n in adj}
    for _ in range(10):
        changed = False
        for n in list(adj):
            counts = {}
            for nb in adj[n]:
                counts[labels[nb]] = counts.get(labels[nb], 0) + 1
            if not counts:
                continue
            best = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
            if best != labels[n]:
                labels[n] = best
                changed = True
        if not changed:
            break
    groups = {}
    for n, lab in labels.items():
        groups.setdefault(lab, []).append(n)
    idx = _paper_index(user_id)
    out = []
    for cid, members in groups.items():
        if len(members) < min_size:
            continue
        out.append({"community": str(cid),
                    "papers": [{"id": m, "title": idx.get(m, {}).get("title", m)}
                               for m in sorted(members)]})
    return sorted(out, key=lambda c: -len(c["papers"]))[:20]


def community_detection(user_id: str, min_size: int = 3):
    """Cluster the topic: communities over the CITES graph."""
    if _have(user_id, "algo.louvain"):
        try:
            r = user_graph(user_id).query(
                "CALL algo.louvain('Paper', 'CITES', {}) YIELD nodeId, communityId "
                "MATCH (p:Paper) WHERE id(p) = nodeId "
                "WITH communityId, collect({id: p.id, title: p.title}) AS papers "
                "WHERE size(papers) >= $min_size "
                "RETURN communityId, papers LIMIT 20",
                params={"min_size": min_size})
            if r.result_set:
                return [{"community": row[0], "papers": row[1]} for row in r.result_set]
        except Exception as e:
            print("community warn:", e)
    return _louvain_py(user_id, min_size)


def influential_in_topic(user_id: str, topic: str, top_k: int = 5):
    """Graph-shaped answer: PageRank restricted to the papers of a topic."""
    ranked = run_pagerank(user_id, top_k=200)
    if _mode == "falkordb":
        try:
            r = user_graph(user_id).query(
                "MATCH (p:Paper)-[:ABOUT]->(t:Topic {name:$t}) "
                "RETURN p.id, p.title ORDER BY p.citations DESC LIMIT 200",
                params={"t": topic})
            ids = {row[0] for row in r.result_set if row[0]}
            hit = [r_ for r_ in ranked if r_["id"] in ids][:top_k]
            if hit:
                return hit
        except Exception:
            pass
    return ranked[:top_k]


def find_experts(user_id: str, query: str, limit: int = 5) -> list:
    """Who knows a topic/method, answered from live graph edges — works on data
    a user actually ingested, not just the seed.

    Primary path: (Person)-[:WROTE]->(Paper)-[:ABOUT]->(Topic). Fallback path:
    (Person)-[:WROTE]->(Paper)-[:USES_METHOD]->(Method). Ranked by distinct papers."""
    q = (query or "").strip()
    if not q:
        return []
    if _mode == "falkordb":
        g = user_graph(user_id)
        for pattern in (
            "MATCH (p:Person)-[:WROTE]->(pa:Paper)-[:ABOUT]->(t:Topic) "
            "WHERE toLower(t.name) CONTAINS toLower($q) "
            "RETURN p.name, count(DISTINCT pa), collect(DISTINCT pa.title)[0..3] "
            "ORDER BY count(DISTINCT pa) DESC LIMIT $lim",
            "MATCH (p:Person)-[:WROTE]->(pa:Paper)-[:USES_METHOD]->(m:Method) "
            "WHERE toLower(m.name) CONTAINS toLower($q) "
            "RETURN p.name, count(DISTINCT pa), collect(DISTINCT pa.title)[0..3] "
            "ORDER BY count(DISTINCT pa) DESC LIMIT $lim",
        ):
            try:
                r = g.query(pattern, params={"q": q[:80], "lim": limit})
                out = [{"name": row[0] or "(unnamed)", "papers": row[1],
                        "examples": [t for t in (row[2] or []) if t]} for row in r.result_set]
                if out:
                    return out
            except Exception as e:
                print("find_experts warn:", e)
        return []
    m = _mem(graph_name(user_id))
    ql = q.lower()
    titles = {p["id"]: p.get("title", p["id"]) for p in m.get("papers", [])}
    counts = {}
    for a in m.get("authored", []):
        pid = a["paper"]
        topic = m.get("paper_topic", {}).get(pid, "")
        meths = " ".join(m.get("methods", {}).get(pid, []))
        if ql in (topic or "").lower() or ql in meths.lower():
            rec = counts.setdefault(a["key"], {"name": a["name"], "papers": 0, "examples": []})
            rec["papers"] += 1
            if len(rec["examples"]) < 3 and titles.get(pid):
                rec["examples"].append(titles[pid])
    return sorted(counts.values(), key=lambda r: -r["papers"])[:limit]


def contradicted_papers(user_id: str) -> set:
    """Paper ids whose claims sit on either end of a CONTRADICTS edge.

    The Critic acts on this set: graph evidence, not keyword vibes."""
    if _mode == "falkordb":
        g = user_graph(user_id)
        out = set()
        for pattern in (
            "MATCH (p:Paper)-[:CLAIMS]->(:Claim)-[:CONTRADICTS]->(:Claim) RETURN DISTINCT p.id",
            "MATCH (p:Paper)-[:CLAIMS]->(:Claim)<-[:CONTRADICTS]-(:Claim) RETURN DISTINCT p.id",
        ):
            try:
                r = g.query(pattern)
                out |= {row[0] for row in r.result_set if row[0]}
            except Exception as e:
                print("contradicted_papers warn:", e)
        return out
    m = _mem(graph_name(user_id))
    claims = {c["id"]: c for c in m.get("claims", [])}
    out = set()
    for l in m.get("claim_links", []):
        if l.get("relation") != "contradicts":
            continue
        for cid in (l["a_id"], l["b_id"]):
            pid = claims.get(cid, {}).get("paper_id")
            if pid:
                out.add(pid)
    return out


def connected_components(user_id: str, min_size: int = 2) -> list:
    """Connected components over undirected CITES: which bodies of work actually touch.
    A citation graph in one giant component means the topic is one conversation."""
    adj = {}
    for a, b in _citations(user_id):
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    seen, groups = set(), []
    for start in adj:
        if start in seen:
            continue
        stack, comp = [start], []
        seen.add(start)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in adj.get(cur, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        if len(comp) >= min_size:
            groups.append(comp)
    idx = _paper_index(user_id)
    groups.sort(key=len, reverse=True)
    return [{"size": len(g), "members": [{"id": i, "title": idx.get(i, {}).get("title", i)}
                                          for i in g[:10]]} for g in groups[:20]]


def _norm_title_local(t: str) -> str:
    return "".join(ch for ch in (t or "").lower() if ch.isalnum())[:60]


def search_papers_by_title(user_id: str, phrase: str, limit: int = 3) -> list:
    """Resolve a paper by normalized title (exact > substring > prefix-score).
    Never invents: returns nothing if nothing in the graph is close."""
    idx = _paper_index(user_id)
    pn = _norm_title_local(phrase)
    if not pn:
        return []
    scored = []
    for pid, meta in idx.items():
        tn = _norm_title_local(meta.get("title") or "")
        if not tn:
            continue
        if tn == pn:
            score = 1.0
        elif pn in tn or tn in pn:
            score = 0.9
        else:
            score = sum(1 for a, b in zip(pn, tn) if a == b) / max(len(pn), len(tn))
        if score >= 0.5:
            scored.append((score, pid, meta))
    scored.sort(key=lambda x: (-x[0], -(x[2].get("citations") or 0)))
    return [{"id": pid, "title": meta.get("title", pid),
             "citations": meta.get("citations", 0)} for _s, pid, meta in scored[:limit]]


def _dfs_paths(adj: dict, start: str, depth: int) -> list:
    """All simple directed paths from start, capped at depth, maximal ones only."""
    out, stack = [], [(start, [start])]
    while stack:
        node, trail = stack.pop()
        nxts = [n for n in adj.get(node, ()) if n not in trail]
        if nxts and len(trail) - 1 < depth:
            for n in nxts:
                stack.append((n, trail + [n]))
        elif len(trail) > 1:
            out.append(trail)
    return out


def trace_lineage(user_id: str, start_ids: list, direction: str = "both",
                  depth: int = 4, max_paths: int = 10) -> list:
    """Bounded directed CITES walks from the given papers.

    direction 'out' = what the paper builds on (follow CITES outward),
    'in' = what builds on the paper (follow CITES inward), 'both' = run each.
    Returns [(label, [{id, title, citations}, ...])] sorted shortest-first,
    then by total citations, capped at max_paths so one hub paper cannot
    flood the panel with every 4-hop detour."""
    ids = [i for i in start_ids if i]
    if not ids:
        return []
    idx = _paper_index(user_id)
    raw = []
    if _mode == "falkordb":
        g = user_graph(user_id)
        proj = "[n IN nodes(path) | {id: n.id, title: n.title, citations: coalesce(n.citations,0)}]"
        patterns = []
        if direction in ("out", "both"):
            patterns.append(("cites",
                             f"MATCH path = (s:Paper)-[:CITES*1..{depth}]->(t:Paper) "
                             f"WHERE s.id IN $ids RETURN {proj} AS chain LIMIT 200"))
        if direction in ("in", "both"):
            patterns.append(("cited-by",
                             f"MATCH path = (s:Paper)<-[:CITES*1..{depth}]-(t:Paper) "
                             f"WHERE s.id IN $ids RETURN {proj} AS chain LIMIT 200"))
        for lbl, q in patterns:
            try:
                r = g.query(q, params={"ids": ids})
                for row in r.result_set:
                    chain = row[0] or []
                    if len(chain) > 1:
                        raw.append((lbl, chain))
            except Exception as e:
                print("trace_lineage warn:", e)
    else:
        adj_out, adj_in = {}, {}
        for a, b in _citations(user_id):
            adj_out.setdefault(a, set()).add(b)
            adj_in.setdefault(b, set()).add(a)
        runs = []
        if direction in ("out", "both"):
            runs.append(("cites", adj_out))
        if direction in ("in", "both"):
            runs.append(("cited-by", adj_in))
        for lbl, adj in runs:
            for sid in ids:
                for trail in _dfs_paths(adj, sid, depth):
                    raw.append((lbl, [{"id": i, "title": idx.get(i, {}).get("title", i),
                                       "citations": idx.get(i, {}).get("citations", 0)}
                                      for i in trail]))
    seen, uniq = set(), []
    for lbl, chain in raw:
        key = (lbl, tuple(c.get("id") for c in chain))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((lbl, chain))
    uniq.sort(key=lambda lc: (len(lc[1]), -sum(c.get("citations") or 0 for c in lc[1])))
    return uniq[:max_paths]


def list_procedures():
    """List available FalkorDB procedures for debugging."""
    if _mode != "falkordb":
        return []
    g = user_graph("test")
    try:
        r = g.query("CALL dbms.procedures() YIELD name RETURN name ORDER BY name")
        return [row[0] for row in r.result_set]
    except Exception as e:
        print("list_procedures warn:", e)
        return []


# ---- Three Memory Types ----

# Episodic: (User)-[:REMEMBERS]->(Episode)-[:MENTIONED]->(Paper|Topic)
def remember_episode(user_id: str, text: str, mentioned_papers: list = None, mentioned_topics: list = None):
    """Store episodic memory with links to mentioned entities.

    The Episode is created, its element id returned, and entity edges are drawn
    from that id: matching an Episode by a second timestamp() call never hits,
    so the old pattern silently dropped every MENTIONED edge."""
    if _mode == "falkordb":
        g = user_graph(user_id)
        try:
            r = g.query(
                "MERGE (u:User {id:$uid}) "
                "CREATE (e:Episode {text:$text, at:timestamp()}) "
                "MERGE (u)-[:REMEMBERS]->(e) RETURN id(e) AS eid",
                params={"uid": sanitize_user_id(user_id), "text": text})
            eid = r.result_set[0][0] if r.result_set else None
            if eid is None:
                return
            for pid in (mentioned_papers or []):
                g.query(
                    "MATCH (e) WHERE id(e)=$eid MATCH (p:Paper {id:$pid}) MERGE (e)-[:MENTIONED]->(p)",
                    params={"eid": eid, "pid": pid})
            for topic in (mentioned_topics or []):
                g.query(
                    "MATCH (e) WHERE id(e)=$eid MERGE (t:Topic {name:$topic}) "
                    "MERGE (e)-[:MENTIONED]->(t)",
                    params={"eid": eid, "topic": topic})
        except Exception as e:
            print("remember_episode warn:", e)
    else:
        m = _mem(graph_name(user_id))
        m.setdefault("episodes", []).append({"text": text, "at": time.time(),
                                             "papers": mentioned_papers or [], "topics": mentioned_topics or []})


# Semantic: (User)-[:INTERESTED_IN {weight, updated_at}]->(Topic)
def update_interest(user_id: str, topic: str, delta: float = 0.1):
    """Update user's interest weight in a topic (positive or negative)."""
    if _mode == "falkordb":
        g = user_graph(user_id)
        try:
            g.query(
                "MERGE (u:User {id:$uid}) MERGE (t:Topic {name:$topic}) "
                "MERGE (u)-[r:INTERESTED_IN]->(t) "
                "SET r.weight = coalesce(r.weight, 0.5) + $delta, r.updated_at = timestamp() "
                "SET r.weight = CASE WHEN r.weight > 1 THEN 1 WHEN r.weight < 0 THEN 0 ELSE r.weight END",
                params={"uid": sanitize_user_id(user_id), "topic": topic, "delta": delta})
        except Exception as e:
            print("update_interest warn:", e)
    else:
        m = _mem(graph_name(user_id))
        m.setdefault("interests", {})
        m["interests"][topic] = max(0, min(1, m["interests"].get(topic, 0.5) + delta))


def get_interests(user_id: str, top_k: int = 10):
    """Get user's top interests."""
    if _mode == "falkordb":
        g = user_graph(user_id)
        try:
            r = g.query(
                "MATCH (u:User {id:$uid})-[r:INTERESTED_IN]->(t:Topic) "
                "RETURN t.name AS topic, r.weight AS weight ORDER BY r.weight DESC LIMIT $k",
                params={"uid": sanitize_user_id(user_id), "k": top_k})
            return [{"topic": row[0], "weight": row[1]} for row in r.result_set]
        except Exception as e:
            print("get_interests warn:", e)
            return []
    m = _mem(graph_name(user_id))
    return [{"topic": k, "weight": v} for k, v in sorted(m.get("interests", {}).items(), key=lambda x: -x[1])[:top_k]]


# Procedural: (Procedure {intent, tool_chain, success_count, created_at})
def store_procedure(user_id: str, intent: str, tool_chain: list, success: bool = True):
    """Store a successful query plan as a reusable procedure."""
    if _mode == "falkordb":
        g = user_graph(user_id)
        try:
            g.query(
                "MERGE (p:Procedure {intent:$intent, tool_chain:$chain}) "
                "ON CREATE SET p.success_count = 1, p.created_at = timestamp() "
                "ON MATCH SET p.success_count = p.success_count + 1, p.last_used = timestamp()",
                params={"uid": sanitize_user_id(user_id), "intent": intent, "chain": str(tool_chain)})
        except Exception as e:
            print("store_procedure warn:", e)
    else:
        m = _mem(graph_name(user_id))
        m.setdefault("procedures", [])
        m["procedures"].append({"intent": intent, "tool_chain": tool_chain, "success": success})


def get_procedures(user_id: str, intent: str = None, limit: int = 5):
    """Retrieve relevant procedures for a given intent."""
    if _mode == "falkordb":
        g = user_graph(user_id)
        try:
            if intent:
                r = g.query(
                    "MATCH (p:Procedure) WHERE p.intent CONTAINS $intent "
                    "RETURN p.intent, p.tool_chain, p.success_count ORDER BY p.success_count DESC LIMIT $lim",
                    params={"intent": intent, "lim": limit})
            else:
                r = g.query(
                    "MATCH (p:Procedure) RETURN p.intent, p.tool_chain, p.success_count ORDER BY p.success_count DESC LIMIT $lim",
                    params={"lim": limit})
            return [{"intent": row[0], "tool_chain": row[1], "success_count": row[2]} for row in r.result_set]
        except Exception as e:
            print("get_procedures warn:", e)
            return []
    m = _mem(graph_name(user_id))
    procs = m.get("procedures", [])
    if intent:
        procs = [p for p in procs if intent.lower() in p["intent"].lower()]
    return sorted(procs, key=lambda x: -x.get("success_count", 1))[:limit]


# ---- Multi-Agent Task Handoff ----

def create_task(user_id: str, from_agent: str, to_agent: str, about_paper: str, status: str = "needs_review", payload: dict = None):
    """Create a Task node for agent handoff. Each Task carries a uid so status
    updates target the exact task, never a positional index that shifts when
    other agents append tasks concurrently."""
    tuid = uuid.uuid4().hex[:12]
    if _mode == "falkordb":
        g = user_graph(user_id)
        try:
            g.query(
                "MERGE (u:User {id:$uid}) "
                "CREATE (t:Task {uid:$tuid, from_agent:$from, to_agent:$to, status:$status, created_at:timestamp(), payload:$payload}) "
                "MERGE (u)-[:OWNS]->(t) "
                "WITH t MATCH (p:Paper {id:$paper}) MERGE (t)-[:ABOUT]->(p)",
                params={"uid": sanitize_user_id(user_id), "tuid": tuid, "from": from_agent,
                        "to": to_agent, "status": status, "paper": about_paper,
                        "payload": str(payload or {})})
        except Exception as e:
            print("create_task warn:", e)
    else:
        m = _mem(graph_name(user_id))
        m.setdefault("tasks", []).append({"uid": tuid, "from_agent": from_agent,
                                          "to_agent": to_agent, "status": status,
                                          "paper": about_paper, "payload": payload})


def get_tasks(user_id: str, status: str = None, to_agent: str = None):
    """Get tasks for an agent."""
    if _mode == "falkordb":
        g = user_graph(user_id)
        try:
            where = []
            params = {"uid": sanitize_user_id(user_id)}
            if status:
                where.append("t.status = $status")
                params["status"] = status
            if to_agent:
                where.append("t.to_agent = $to_agent")
                params["to_agent"] = to_agent
            where_clause = "WHERE " + " AND ".join(where) if where else ""
            r = g.query(
                f"MATCH (u:User {{id:$uid}})-[:OWNS]->(t:Task) {where_clause} "
                "OPTIONAL MATCH (t)-[:ABOUT]->(p:Paper) "
                "RETURN t.uid, t.from_agent, t.to_agent, t.status, t.payload, p.id, p.title "
                "ORDER BY t.created_at DESC LIMIT 20",
                params=params)
            return [{"uid": row[0], "from": row[1], "to": row[2], "status": row[3],
                     "payload": row[4], "paper_id": row[5], "paper_title": row[6]}
                    for row in r.result_set]
        except Exception as e:
            print("get_tasks warn:", e)
            return []
    m = _mem(graph_name(user_id))
    tasks = m.get("tasks", [])
    if status:
        tasks = [t for t in tasks if t["status"] == status]
    if to_agent:
        tasks = [t for t in tasks if t["to_agent"] == to_agent]
    # newest-first, same field shape as the falkordb branch so callers work in both modes
    out = []
    for t in reversed(tasks):
        d = dict(t)
        d.setdefault("paper_id", d.get("paper", ""))
        d.setdefault("from", d.get("from_agent"))
        d.setdefault("to", d.get("to_agent"))
        out.append(d)
    return out


def update_task_status(user_id: str, task_uid: str, status: str):
    """Update one task's status by uid."""
    if _mode == "falkordb":
        g = user_graph(user_id)
        try:
            g.query(
                "MATCH (u:User {id:$uid})-[:OWNS]->(t:Task {uid:$tuid}) "
                "SET t.status = $status, t.updated_at = timestamp()",
                params={"uid": sanitize_user_id(user_id), "tuid": task_uid, "status": status})
        except Exception as e:
            print("update_task_status warn:", e)
    else:
        for t in _mem(graph_name(user_id)).get("tasks", []):
            if t.get("uid") == task_uid:
                t["status"] = status
                return
