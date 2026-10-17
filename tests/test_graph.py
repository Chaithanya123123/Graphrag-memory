"""Graph algorithms and the three memory types, run against the in-memory backend."""
import app.graph as G


def _seed(uid="algo"):
    G._MEM.clear()
    m = G._mem(G.graph_name(uid))
    m["papers"] = [
        {"id": "a", "title": "A", "year": 2020, "citations": 5},
        {"id": "b", "title": "B", "year": 2021, "citations": 10},
        {"id": "c", "title": "C", "year": 2022, "citations": 50},
        {"id": "d", "title": "D", "year": 2023, "citations": 1},
        {"id": "island", "title": "Isolated", "year": 2024, "citations": 0},
    ]
    m["cites"] = [
        {"from": "a", "to": "b"},   # a -> b
        {"from": "b", "to": "c"},   # b -> c
        {"from": "a", "to": "c"},   # a -> c (shortcut)
    ]
    return uid


def test_pagerank_ranks_cited_paper_higher():
    uid = _seed()
    ranked = G.run_pagerank(uid, top_k=10)
    order = [r["id"] for r in ranked]
    assert "c" in order
    assert order.index("c") < order.index("b") < order.index("a")
    assert ranked[0]["score"] > 0


def test_pagerank_scores_form_distribution():
    uid = _seed()
    ranked = G.run_pagerank(uid, top_k=10)
    assert abs(sum(r["score"] for r in ranked) - 1.0) < 1e-6


def test_shortest_path_uses_fewest_hops():
    uid = _seed()
    path = G.shortest_path_between(uid, "a", "c", 6)
    assert [n["id"] for n in path] == ["a", "c"], "should take the 1-hop shortcut, not a->b->c"


def test_shortest_path_respects_direction_of_query():
    uid = _seed()
    # undirected traversal: a path exists in reverse too
    path = G.shortest_path_between(uid, "c", "a", 6)
    assert [n["id"] for n in path] == ["c", "a"]


def test_shortest_path_unknown_ids_returns_empty():
    uid = _seed()
    assert G.shortest_path_between(uid, "a", "nope", 6) == []


def test_path_to_isolated_node_is_empty():
    uid = _seed()
    assert G.shortest_path_between(uid, "a", "island", 6) == []


def test_community_detection_groups_connected_papers():
    uid = _seed()
    comms = G.community_detection(uid, min_size=2)
    assert comms, "no communities found"
    members = {p["id"] for c in comms for p in c["papers"]}
    assert {"a", "b", "c"} <= members
    assert "island" not in members


def test_semantic_memory_accumulates_and_clamps():
    uid = _seed("mem")
    G.update_interest(uid, "graph rag", 0.1)
    w1 = G.get_interests(uid)[0]["weight"]
    G.update_interest(uid, "graph rag", 0.1)
    w2 = G.get_interests(uid)[0]["weight"]
    assert w2 > w1
    G.update_interest(uid, "graph rag", 10.0)
    assert G.get_interests(uid)[0]["weight"] <= 1.0


def test_episodic_memory_records_mentions():
    uid = _seed("epi")
    G.remember_episode(uid, "who wrote the survey?", mentioned_papers=["a"], mentioned_topics=["graph rag"])
    G.remember_episode(uid, "anything else?", mentioned_papers=[], mentioned_topics=[])
    eps = G._mem(G.graph_name(uid))["episodes"]
    assert len(eps) == 2
    assert eps[0]["papers"] == ["a"] and eps[0]["topics"] == ["graph rag"]


def test_procedural_memory_retrieves_similar_intent():
    uid = _seed("proc")
    G.store_procedure(uid, intent="find contradictions in graph rag", tool_chain=["find_contradictions"])
    G.store_procedure(uid, intent="most influential paper", tool_chain=["pagerank"])
    hits = G.get_procedures(uid, intent="contradictions")
    assert hits and "contradictions" in hits[0]["intent"]


def test_supersede_keeps_history():
    uid = _seed("sup")
    G.store_claim_nodes(uid, "a", "Paper A", "topic", [{"id": "c1", "text": "The sky is blue"}], "S2")
    G.store_claim_nodes(uid, "b", "Paper B", "topic", [{"id": "c2", "text": "the sky is blue"}], "S2")
    sup = G._mem(G.graph_name(uid)).get("superseded", [])
    assert any(s["from"] == "c1" and s["to"] == "c2" for s in sup)
    texts = {c["id"]: c["text"] for c in G._mem(G.graph_name(uid))["claims"]}
    assert texts["c1"] == "The sky is blue", "old claim must not be overwritten"


def test_tenant_graphs_are_separate():
    G._MEM.clear()
    G._seed_ = None
    _seed("t1")
    G.store_procedure("t1", intent="only-for-t1", tool_chain=["x"])
    assert G.get_procedures("t2") == []