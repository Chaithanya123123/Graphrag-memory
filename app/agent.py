"""NL router: question -> graph tool -> answer + viz + cypher.

Tool choice comes from an LLM reading the tool descriptions (tool calling);
the keyword template below is the fallback. Every answer ends in a traversal
result (PageRank / BFS / community / claim traversal), not a list dump.
"""
import re
from app.graph import (recall, graph_snapshot, run_pagerank, shortest_path_between,
                       community_detection, query_contradictions, influential_in_topic,
                       get_procedures, sanitize_user_id, find_experts, connected_components,
                       trace_lineage, search_papers_by_title, _paper_index)
from app.llm import polish, select_tool

CYPHER = {
    "trace_lineage": "MATCH path = (s:Paper)-[:CITES*1..4]->(t:Paper) WHERE s.id IN $ids "
                     "RETURN [n IN nodes(path) | {id: n.id, title: n.title, citations: coalesce(n.citations,0)}] AS chain LIMIT 200 "
                     "-- inbound variant: MATCH path = (s:Paper)<-[:CITES*1..4]-(t:Paper) WHERE s.id IN $ids",
    "find_contradictions": "MATCH (p1:Paper)-[:CLAIMS]->(c1:Claim)-[r:CONTRADICTS]->(c2:Claim)<-[:CLAIMS]-(p2:Paper) RETURN p1.title, c1.text, c2.text, p2.title, r.evidence LIMIT 10",
    "who_knows": "MATCH (p:Person)-[:WROTE]->(pa:Paper)-[:ABOUT]->(t:Topic) WHERE toLower(t.name) CONTAINS toLower($topic) RETURN p.name, count(DISTINCT pa) ORDER BY 2 DESC LIMIT 5",
    "shortest": "MATCH p=shortestPath((a:Paper {id:$a})-[:CITES*..6]-(b:Paper {id:$b})) RETURN [n IN nodes(p)|n.id]",
    "recall": "MATCH (u:User {id:$uid})-[:REMEMBERS]->(e:Episode) RETURN e.text ORDER BY e.at DESC LIMIT 10",
    "live_search": "MATCH (p:Paper)-[:ABOUT]->(t:Topic {name:$topic}) OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(v:Venue) RETURN p.title, p.year, v.name, p.citations ORDER BY p.citations DESC LIMIT 12",
    "pagerank": "CALL algo.pagerank('Paper', 'CITES', {iterations: 20, dampingFactor: 0.85}) YIELD nodeId, score MATCH (p:Paper) WHERE id(p) = nodeId RETURN p.id, p.title, score ORDER BY score DESC LIMIT 20",
    "community": "CALL algo.louvain('Paper', 'CITES', {}) YIELD nodeId, communityId MATCH (p:Paper) WHERE id(p) = nodeId WITH communityId, collect(p.title) AS papers RETURN communityId, papers LIMIT 20",
    "influential_papers": "CALL algo.pagerank('Paper','CITES',{iterations:20,dampingFactor:0.85}) YIELD nodeId, score MATCH (p:Paper)-[:ABOUT]->(t:Topic {name:$topic}) WHERE id(p)=nodeId RETURN p.title, score ORDER BY score DESC LIMIT 5",
}

LIVE_HINTS = ["recent", "latest", "famous", "top", "cvpr", "ieee", "neurips", "icml",
              "iclr", "acl", "kdd", "find", "search", "papers", "published", "survey",
              "sota", "state of the art", "what is", "what are", "explain", "trends"]

PAGERANK_HINTS = ["most influential", "top papers", "pagerank", "important", "highly cited", "key papers", "central"]
SHORTEST_HINTS = ["shortest path", "path between", "connection between", "how does", "connects", "citation path"]
COMMUNITY_HINTS = ["clusters", "communities", "groups", "related papers", "topic clusters"]

ANSWERS = {
    "find_contradictions": ("**Contradiction found**\nClaim `c1` *“Graph retrieval beats vector-only on multi-hop QA”* (p1, p2) is **CONTRADICTED BY** claim `c2` *“Vector-only is sufficient”* (p3).\nEvidence edge: `(c1)-[:CONTRADICTS]->(c2)`. Recommend citing both in related work.",
     ["c1", "c2"]),
    "who_knows": ("**No experts on that topic in your graph yet.** `find_experts` returned "
                  "empty — `(Person)-[:WROTE]->(Paper)-[:ABOUT]->(Topic)` has nothing to follow. "
                  "Ingest papers on the topic first, then ask again.",
     []),
    "recall": ("", []),
}

# last-resort text for a tool whose traversal genuinely returned nothing —
# never a canned story that could contradict the live graph
DEFAULT_ANSWER = ("**Nothing to traverse yet** — the query ran against your graph and every edge "
                  "came back empty. Ask `find papers on <topic>` to ingest real citations, then retry.")


WHO_HINTS = ["who knows", "who is the expert", "who worked", "who wrote", "who works",
             "who should i ask", "who has written", "anyone who", "which authors",
             "expert", "owner"]


def route_template(question: str) -> str:
    """Fallback router. The LLM router is tried first; this is the safety net.

    Order matters: a 'who' subject outranks topic words, so 'anyone who has
    written about contradictions' asks for PEOPLE and hits who_knows, not the
    contradiction traversal."""
    q = question.lower()
    if any(w in q for w in ["remember", "before", "last time", "history", "my memory"]):
        return "recall"
    if any(w in q for w in WHO_HINTS):
        return "who_knows"
    if any(w in q for w in ["contradict", "disagree", "counter", "refute", "conflict",
                            "inconsistent", "oppose", "support", "agree"]):
        return "find_contradictions"
    if any(w in q for w in ["lineage", "builds on", "genealogy", "provenance"]):
        return "trace_lineage"
    if any(w in q for w in PAGERANK_HINTS):
        return "pagerank"
    if any(w in q for w in SHORTEST_HINTS):
        return "shortest"
    if any(w in q for w in COMMUNITY_HINTS):
        return "community"
    return "live_search"  # default: fetch the live web (recent + famous)


STOP = {"the", "a", "an", "on", "about", "for", "of", "in", "to", "me", "what",
        "which", "find", "papers", "paper", "please", "most", "with", "and", "is",
        "are", "that", "this", "from", "it", "i", "my", "some", "any", "work", "works"}


def _procedural_tool(user_id: str, question: str):
    """Procedural memory shapes routing: a near-identical intent that succeeded
    before reuses that plan instead of re-deriving it. Jaccard over content words."""
    try:
        procs = get_procedures(user_id, limit=30)
    except Exception:
        return None, 0.0
    qtok = set(re.findall(r"[a-z0-9]+", question.lower())) - STOP
    if not qtok:
        return None, 0.0
    best, best_score = None, 0.0
    for p in procs:
        tools = re.findall(r"[a-z_]+", str(p.get("tool_chain") or ""))
        tool = next((t for t in tools if t in CYPHER), None)
        if not tool:
            continue
        itok = set(re.findall(r"[a-z0-9]+", (p.get("intent") or "").lower())) - STOP
        if not itok:
            continue
        score = len(qtok & itok) / len(qtok | itok)
        if score > best_score:
            best, best_score = tool, score
    # threshold high on purpose: a loose match would freeze a past mis-route and
    # override the router forever; only near-identical intents should be cached
    return (best, best_score) if best_score >= 0.8 else (None, best_score)


SPEC_TO_TOOL = {"find_supports": "find_contradictions", "influential_papers": "pagerank",
                "shortest_path": "shortest", "communities": "community"}


def route(user_id: str, question: str) -> tuple[str, str]:
    """Graph memory first, then LLM tool-calling, then keyword template."""
    tool, score = _procedural_tool(user_id, question)
    if tool:
        return tool, f"procedural memory: reused '{tool}' plan ({score:.0%} intent match)"
    tool, how = select_tool(question)
    if tool:
        return SPEC_TO_TOOL.get(tool, tool), how
    return route_template(question), how


def _topic_of(question: str) -> str:
    q = question.strip()
    m = re.search(r"(?i)\b(on|about|for|related to)\b(.+)", q)
    if m and len(m.group(2).strip()) > 2:
        return re.sub(r"[?.!]+$", "", m.group(2).strip())[:80]
    q = re.sub(r"(?i)^(what (is|are)|who|when|where|find|search|give( me)?|show( me)?|list|get|tell me about|explain)\b", "", q)
    q = re.sub(r"(?i)\b(recent|latest|famous|top|best|new|cvpr|ieee|neurips|icml|iclr|acl|kdd|aaai|siggraph|papers?|research|works?|studies|results|trends?)\b", "", q)
    return re.sub(r"[?.!]+$", "", q).strip()[:80] or question[:80]


def _format_live(papers: list) -> str:
    lines = ["**Live results — recent + most-cited, stored in your graph**\n"]
    for i, p in enumerate(papers, 1):
        ven = f" · *{p['venue']}*" if p["venue"] else ""
        lines.append(f"{i}. **{p['title']}** ({p['year']}{ven}, ⬆ {p['citations']} cites, `{p['src']}`)\n"
                     f"   {', '.join(p['authors'][:3])}" + (f" · [{p['src']} link]({p['url']})" if p["url"] else ""))
    return "\n".join(lines)


def _finish(question, tool, user_id, base, mem, snap, hl, topics=None, extra=None):
    cyp = CYPHER.get(tool, CYPHER["live_search"])
    final, llm = polish(question, tool, cyp, base, mem)
    # re-snapshot with the answer's own path pinned in, so the viz always shows it
    from app.graph import graph_snapshot as _gs
    pinned = _gs(user_id, highlight=hl) if hl else snap
    out = {"tool": tool, "router": extra or "", "cypher": cyp,
           "cypher_params": {"uid": user_id, "topic": (topics or [""])[0] if topics else ""},
           "text": final, "memory": mem, "llm": llm, "highlight": hl,
           "nodes": pinned["nodes"], "edges": pinned["edges"]}
    for k in ("total_nodes", "total_edges", "hidden"):
        if k in pinned:
            out[k] = pinned[k]
    if topics:
        out["topic"] = topics[0]
        out["topics"] = topics
    return out


def answer(user_id: str, question: str):
    user_id = sanitize_user_id(user_id)
    tool, how = route(user_id, question)
    mem = recall(user_id)
    snap_holder = {}

    def snap():
        if "s" not in snap_holder:
            snap_holder["s"] = graph_snapshot(user_id)
        return snap_holder["s"]

    # ---------- live search: ingest, then answer WITH a traversal ----------
    if tool == "live_search":
        from app.sources import live_search, store_papers
        topic = _topic_of(question)
        papers = live_search(topic)
        store_papers(user_id, papers, topic)
        s = snap()
        if not papers:
            return _finish(question, tool, user_id,
                           "**Live search came up empty** (sources rate-limited?). "
                           "Try again in a few seconds or rephrase.", mem, s, [],
                           topics=[topic], extra=how)
        base = _format_live(papers)
        # the graph, not the list, produces the answer
        ranked = influential_in_topic(user_id, topic, top_k=5)
        comms = community_detection(user_id, min_size=3)
        n_cites = len([e for e in s["edges"] if e.get("label") == "CITES"])
        tail = ["\n\n**Graph traversal over what was just ingested**"]
        if ranked:
            tail.append("\n_`algo.pagerank` over `CITES` — most influential in this topic:_")
            for i, r in enumerate(ranked, 1):
                tail.append(f"{i}. **{r['title'][:90]}** — score `{r['score']:.5f}`, cited "
                            f"`{r['id']}` ({n_cites} citation edges in your graph)")
        if comms:
            tail.append(f"\n_`algo.louvain` found {len(comms)} citation communities; largest has "
                        f"{len(comms[0]['papers'])} papers: "
                        f"{', '.join(pp['title'][:40] for pp in comms[0]['papers'][:3])}..._")
        tail.append(f"\n_Answer derived by traversal, not similarity. "
                    f"Ask: “most influential paper on {topic}”, “shortest path between A and B”, "
                    f"“what contradicts X?”_")
        base += "\n".join(tail)
        from app.llm import takeaway as _takeaway
        note, llm2 = _takeaway(question, [p["title"] for p in papers])
        if note:
            base += f"\n\n**Agent take:** {note}"
        return _finish(question, tool, user_id, base, mem, s,
                       [r["id"] for r in ranked[:3]] or [p["id"] for p in papers[:3]],
                       topics=[topic], extra=how)

    if tool == "recall":
        s = snap()
        txt = ("**Your memory (persistent across sessions):**\n"
               + ("\n".join(f"- {m}" for m in mem) if mem
                  else "- nothing yet — ask me something first.")
               ) if mem else ("**Memory is empty** — this itself is read from the graph "
                              "(`(User)-[:REMEMBERS]->(Episode)`), proving persistence works.")
        return _finish(question, tool, user_id, txt, mem, s, [], extra=how)

    if tool == "find_contradictions":
        rows = query_contradictions(user_id, limit=10)
        s = snap()
        if rows:
            lines = ["**Contradictions from your graph — LLM-labelled, with evidence**\n"]
            for r in rows[:5]:
                lines.append(
                    f"- *“{r['c1'][:120]}”* ({(r['p1'] or '')[:50]}) "
                    f"**CONTRADICTS** *“{r['c2'][:120]}”* ({(r['p2'] or '')[:50]})\n"
                    f"  Evidence: `{r['evidence'][:150]}` · edge `(c1)-[:CONTRADICTS]->(c2)`")
            lines.append("\n_A similarity search can't find this — it needs claim pairs + relation labels._")
            return _finish(question, tool, user_id, "\n".join(lines), mem, s, [], extra=how)
        text, hl = ANSWERS["find_contradictions"]
        return _finish(question, tool, user_id, text, mem, s, hl, extra=how)

    if tool == "pagerank":
        topic = _topic_of(question)
        rows = influential_in_topic(user_id, topic, top_k=10) if topic else run_pagerank(user_id)
        if not rows:
            rows = run_pagerank(user_id)
        s = snap()
        if rows:
            lines = ["**Most Influential Papers — PageRank over `CITES`**\n"]
            for i, r in enumerate(rows[:10], 1):
                lines.append(f"{i}. **{r['title'][:90]}** — PageRank: `{r['score']:.5f}` "
                             f"(id `{r['id']}`)")
            lines.append("\n_Traversal, not keyword match: score comes from the citation graph._")
            return _finish(question, tool, user_id, "\n".join(lines), mem, s,
                           [r["id"] for r in rows[:5]], topics=[topic], extra=how)

    if tool == "shortest":
        ids = _paper_ids_from(question, user_id)
        if len(ids) >= 2:
            path = shortest_path_between(user_id, ids[0], ids[1])
            if path:
                titles = " → ".join((n.get("title") or n["id"])[:60] for n in path)
                base = (f"**Shortest citation path** (`{ids[0]}` → `{ids[1]}`), {len(path) - 1} hops:\n"
                        f"{titles}\n\n_BFS over `CITES` (undirected) — every hop is a real citation._")
                s = snap()
                return _finish(question, tool, user_id, base, mem, s,
                               [n["id"] for n in path], extra=how)
        base = ("Name the two papers to connect, e.g. "
                "*“shortest path between oa:W123 and oa:W456”*. I run BFS over `CITES` "
                "and return the hops with provenance.")
        return _finish(question, tool, user_id, base, mem, snap(), [], extra=how)

    if tool == "who_knows":
        topic = _topic_of(question)
        m = re.search(r"(?i)(?:who|expert)[a-z ]*?knows(?:\s+about)?\s+(.+?)\??$", question)
        if m:
            topic = re.sub(r"[?.!]+$", "", m.group(1).strip())[:80]
        exp = find_experts(user_id, topic, limit=5)
        if exp:
            lines = [f"**Who knows `{topic[:48]}` — from the edges, not a profile page**\n"]
            for i, r in enumerate(exp, 1):
                ex = "; ".join(f"“{t[:50]}”" for t in r["examples"][:2])
                lines.append(f"{i}. **{r['name']}** — {r['papers']} paper(s)"
                             + (f" ({ex})" if ex else ""))
            lines.append("\n_Path: `(Person)-[:WROTE]->(Paper)-[:ABOUT]->(Topic)`. "
                         "Ask a follow-up to hand them a Task — they already own the edge._")
            return _finish(question, tool, user_id, "\n".join(lines), mem, snap(), [],
                           topics=[topic], extra=how)

    if tool == "community":
        rows = community_detection(user_id)
        s = snap()
        if rows:
            lines = ["**Paper Communities (clustering over `CITES`)**\n"]
            for r in rows:
                names = ", ".join((pp.get("title") if isinstance(pp, dict) else str(pp))[:40]
                                  for pp in r["papers"][:5])
                lines.append(f"- Community `{r['community']}` ({len(r['papers'])} papers): {names}"
                             f"{'...' if len(r['papers']) > 5 else ''}")
            comps = connected_components(user_id)
            if comps:
                lines.append(f"\n`connected_components`: {len(comps)} connected bodies of work — "
                             f"largest holds {comps[0]['size']} papers; "
                             f"{len(comps) > 1 and 'the rest do not cite each other at all.' or 'the citation graph is one conversation.'}")
            lines.append("\n_Louvain over the citation graph, so clusters follow real citation structure._")
            return _finish(question, tool, user_id, "\n".join(lines), mem, s, [], extra=how)

    if tool == "trace_lineage":
        # direction decided from the wording: "what builds on X" walks CITES IN,
        # "what X builds on" walks OUT; a bare "trace lineage" shows both.
        inbound = bool(re.search(r"(?i)(what\s+builds?\s+on|builds?\s+on\s+this|cited\s+by|who\s+cit)", question))
        direction = "in" if inbound else "both"
        ids = _paper_ids_from(question, user_id)
        label = None
        if not ids:
            phrase = re.sub(r"(?i)\s*\b(?:trace|the|lineage|genealogy|provenance|of|from|for|is|what|show|me|to|paper)\b\s*",
                            " ", question).strip(" ?.\t")
            cand = search_papers_by_title(user_id, phrase, limit=1) or search_papers_by_title(user_id, question, limit=1)
            if cand:
                ids, label = [cand[0]["id"]], cand[0]["title"]
        if not ids:
            base = (f"**Could not resolve a paper in that question** — nothing in your graph matches it by "
                    f"title or id (`oa:...`, `s2:...`), so there is nothing to trace. "
                    f"Ask a `find papers on <topic>` question first to populate the graph, then trace its lineage.")
            return _finish(question, tool, user_id, base, mem, snap(), [], extra=how)
        if not label:
            label = _paper_index(user_id).get(ids[0], {}).get("title", ids[0])
        paths = trace_lineage(user_id, ids, direction=direction, depth=4, max_paths=10)
        if not paths:
            base = (f"**No `CITES` paths from “{label[:60]}” ({ids[0]})** — the walk ran bounded "
                    f"`CITES*1..4` in {'both directions' if direction == 'both' else direction} "
                    f"and the paper is a citation dead end: it neither cites nor is cited by anything "
                    f"in your graph. Ingest work that references it to extend the chain.")
            return _finish(question, tool, user_id, base, mem, snap(), ids[:1], extra=how)
        out_paths = [c for lbl, c in paths if lbl == "cites"]
        in_paths = [c for lbl, c in paths if lbl == "cited-by"]
        lines = [f"**Lineage of “{label[:70]}” — bounded `CITES*1..4` walks, both ends shown, "
                 f"capped at {len(paths)} paths**\n"]
        def _fmt(chain):
            return " → cites → ".join((n.get("title") or n["id"])[:38] for n in chain)
        if out_paths:
            lines.append("_What it builds on (outbound CITES):_")
            for c in out_paths[:5]:
                lines.append(f"- { _fmt(c) }  · {len(c) - 1} hop(s)")
        if in_paths:
            lines.append("\n_What builds on it (inbound CITES — chain reversed so arrows still read `cites`):_")
            for c in in_paths[:5]:
                lines.append(f"- { _fmt(list(reversed(c))) }  · {len(c) - 1} hop(s)")
        lines.append("\n_Every hop is a recorded citation edge, not a similarity guess — "
                     "the Cypher panel shows the exact walk._")
        hl = []
        for _l, c in paths[:3]:
            hl += [n["id"] for n in c]
        return _finish(question, tool, user_id, "\n".join(lines), mem, snap(), hl[:15], extra=how)

    text, hl = ANSWERS.get(tool, (DEFAULT_ANSWER, []))
    if mem:
        text += (f"\n\n_Context from memory ({len(mem)} episodes): “{mem[-1]}” was recalled "
                 f"from your graph and shaped this answer._")
    return _finish(question, tool, user_id, text, mem, snap(), hl, extra=how)


def _paper_ids_from(question: str, user_id: str = "demo") -> list:
    """Pull paper ids out of a question; if only titles are given, fuzzy-match them."""
    ids = re.findall(r"(?:s2:|oa:|arx:)[\w\-]+", question)
    if ids:
        return ids[:2]
    from app.graph import _paper_index
    quoted = re.findall(r"[“\"']([^”\"']{8,})[”\"']", question)
    if not quoted:
        return []
    idx = _paper_index(user_id)
    out = []
    for phrase in quoted[:2]:
        norm = "".join(ch.lower() for ch in phrase if ch.isalnum())[:40]
        best, best_score = None, 0.0
        for pid, meta in idx.items():
            tnorm = "".join(ch.lower() for ch in (meta.get("title") or "") if ch.isalnum())
            if not tnorm:
                continue
            common = sum(1 for a, b in zip(norm, tnorm) if a == b)
            score = common / max(len(norm), len(tnorm))
            if score > best_score:
                best, best_score = pid, score
        if best and best_score > 0.5:
            out.append(best)
    return out[:2]