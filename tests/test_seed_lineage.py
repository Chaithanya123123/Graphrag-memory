"""Seed + lineage tests.

Seed edges must exist on BOTH backends — the FalkorDB half runs whenever a live
FalkorDB answers on 6379 (skip otherwise), because every recent production bug
here was the memory fallback passing while the real DB drifted.
"""
import os
import uuid

import pytest

import app.graph as G

try:
    from falkordb import FalkorDB
    _probe = FalkorDB(host=os.getenv("FALKOR_HOST", "localhost"),
                      port=int(os.getenv("FALKOR_PORT", "6379")))
    _probe.list_graphs()
    _FALKOR = True
except Exception:
    _FALKOR = False

needs_falkordb = pytest.mark.skipif(not _FALKOR, reason="needs a live FalkorDB on 6379")


@pytest.fixture
def fresh_user():
    uid = "seed" + uuid.uuid4().hex[:10]
    yield uid
    if _FALKOR:
        try:
            _probe.delete_graph(G.graph_name(uid))
        except Exception:
            pass


# ---------- memory mode (always runs) ----------

def test_memory_seed_edges_all_present():
    G.init_graph("s_mem1")
    snap = G._snapshot_all("s_mem1")
    labels = {e["label"] for e in snap["edges"]}
    for rel in ("WROTE", "USES_METHOD", "WORKS_ON", "ABOUT", "CITES", "CONTRADICTS", "CLAIMS"):
        assert rel in labels, f"seed missing {rel}"


def test_memory_seed_idempotent():
    G.init_graph("s_mem2")
    a = G._snapshot_all("s_mem2")
    G.init_graph("s_mem2")
    G.init_graph("s_mem2")
    b = G._snapshot_all("s_mem2")
    assert len(a["nodes"]) == len(b["nodes"])
    assert len(a["edges"]) == len(b["edges"])


def test_seed_chain_lineage_3_papers():
    """p2 -> p1 -> p3 exists in the seed: tracing p2 must surface a 2-hop chain."""
    G.init_graph("s_mem3")
    paths = G.trace_lineage("s_mem3", ["p2"], direction="out", depth=4)
    chains = [tuple(n["id"] for n in c) for _l, c in paths]
    assert ("p2", "p1", "p3") in chains, f"seed chain missing: {chains}"
    assert len(paths[0][1]) >= 3


def test_lineage_direction_inbound():
    G.init_graph("s_mem4")
    _label, chain = G.trace_lineage("s_mem4", ["p1"], direction="in", depth=4)[0]
    assert chain[0]["id"] == "p1"          # starts at the traced paper
    assert "p2" in {n["id"] for n in chain}  # FalkorDB Memory cites it

def test_search_papers_by_title_exact_and_no_match():
    G.init_graph("s_mem5")
    hit = G.search_papers_by_title("s_mem5", "graph rag survey", limit=3)
    assert hit and hit[0]["id"] == "p1"
    assert G.search_papers_by_title("s_mem5", "quantum tea brewing", limit=3) == []


def test_agent_lineage_no_match_is_explicit():
    from app.agent import answer
    res = answer("s_mem6", "trace lineage of Quantum Tea Brewing 2099")
    assert res["tool"] == "trace_lineage"
    assert "Could not resolve" in res["text"] or "No `CITES` paths" in res["text"]
    assert "Lineage traced via" not in res["text"]  # old canned story must be dead


def test_agent_lineage_finds_seed_chain():
    from app.agent import answer
    G.init_graph("s_mem7")
    res = answer("s_mem7", "trace lineage of Graph RAG Survey")
    assert res["tool"] == "trace_lineage"
    assert "cites" in res["text"]
    assert res["highlight"], "lineage answer must pin the chain in the viz"


def test_who_knows_seed_edges_live():
    """The README demo question must answer from edges, not canned text, in BOTH modes."""
    from app.agent import answer
    G.init_graph("s_mem8")
    res = answer("s_mem8", "Who knows about GraphRAG?")
    assert res["tool"] == "who_knows"
    assert "Chaithanya" in res["text"]


# ---------- FalkorDB mode (runs when the DB is up) ----------

@needs_falkordb
def test_falkordb_seed_edges_exist(fresh_user, monkeypatch):
    monkeypatch.setattr(G, "_db", _probe)
    monkeypatch.setattr(G, "_mode", "falkordb")
    G.init_graph(fresh_user)
    g = _probe.select_graph(G.graph_name(fresh_user))
    counts = {row[0]: row[1] for row in
              g.query("MATCH ()-[e]->() RETURN type(e), count(e)").result_set}
    for rel in ("WROTE", "USES_METHOD", "WORKS_ON", "ABOUT", "CITES", "CONTRADICTS", "CLAIMS"):
        assert counts.get(rel, 0) >= 1, f"seed missing {rel}: {counts}"


@needs_falkordb
def test_falkordb_init_is_idempotent(fresh_user, monkeypatch):
    monkeypatch.setattr(G, "_db", _probe)
    monkeypatch.setattr(G, "_mode", "falkordb")
    G.init_graph(fresh_user)
    g = _probe.select_graph(G.graph_name(fresh_user))
    n1 = g.query("MATCH (n) RETURN count(n)").result_set[0][0]
    e1 = g.query("MATCH ()-[r]->() RETURN count(r)").result_set[0][0]
    G.init_graph(fresh_user)
    n2 = g.query("MATCH (n) RETURN count(n)").result_set[0][0]
    e2 = g.query("MATCH ()-[r]->() RETURN count(r)").result_set[0][0]
    assert (n2, e2) == (n1, e1), f"second init changed counts: {n1}/{e1} -> {n2}/{e2}"


@needs_falkordb
def test_falkordb_who_knows_answers_from_seed(fresh_user, monkeypatch):
    monkeypatch.setattr(G, "_db", _probe)
    monkeypatch.setattr(G, "_mode", "falkordb")
    G.init_graph(fresh_user)
    exp = G.find_experts(fresh_user, "GraphRAG", limit=5)
    assert exp and exp[0]["name"] == "Chaithanya"


@needs_falkordb
def test_falkordb_lineage_chain(fresh_user, monkeypatch):
    monkeypatch.setattr(G, "_db", _probe)
    monkeypatch.setattr(G, "_mode", "falkordb")
    G.init_graph(fresh_user)
    paths = G.trace_lineage(fresh_user, ["p2"], direction="out", depth=4)
    chains = [tuple(n["id"] for n in c) for _l, c in paths]
    assert ("p2", "p1", "p3") in chains
