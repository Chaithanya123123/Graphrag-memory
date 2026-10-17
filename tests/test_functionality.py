"""Feature tests for the scored brief: live experts, memory-shaped routing,
graph-driven critic, connected components."""
import app.graph as G
from app.agent import answer, route
from app.agents import critic_review
from app.sources import store_papers


def _papers():
    return [
        {"id": "oa:AAA", "title": "Graph RAG Survey", "year": 2024, "venue": "ACM",
         "citations": 120, "authors": ["Ada Lovelace"],
         "abstract": "", "url": "", "src": "OpenAlex", "referenced_works": [],
         "author_keys": ["oa:A1"]},
    ]


def test_who_knows_answers_from_edges(papers):
    store_papers("u1", papers, "retrieval augmented generation")
    res = answer("u1", "who knows about retrieval augmented generation")
    assert res["tool"] == "who_knows"
    assert "Ada Lovelace" in res["text"]
    assert "ABOUT" in res["cypher"]


def test_who_knows_empty_graph_says_so():
    res = answer("u2", "who knows about distributed consensus")
    assert res["tool"] == "who_knows"
    assert "No experts" in res["text"] or "WROTE" in res["text"]


def test_procedural_memory_shapes_routing():
    G.store_procedure("u3", intent="most influential paper on graph rag",
                      tool_chain=["pagerank"], success=True)
    tool, how = route("u3", "most influential paper on graph rag")
    assert tool == "pagerank"
    assert "procedural memory" in how


def test_procedural_memory_does_not_freeze_nearby_intents():
    """'who knows about X' must NOT be hijacked by a stored live_search plan on
    topic X — a loose threshold would cache a past mis-route forever."""
    G.store_procedure("u3b", intent="find papers on graph retrieval",
                      tool_chain=["live_search"], success=True)
    tool, _ = route("u3b", "who knows about graph retrieval")
    assert tool != "live_search"


def test_contradicted_papers_from_real_links():
    G.store_claim_nodes("u4", "oa:XYZ", "Some Paper", "topic",
                        [{"id": "cA", "text": "Method X beats Y on every benchmark we tried this year"},
                         {"id": "cB", "text": "Method Y is sufficient and X adds no benefit at all"}], "test")
    G.store_claim_links("u4", [{"a_id": "cA", "b_id": "cB", "relation": "contradicts",
                                "evidence": "direct opposite"}])
    assert "oa:XYZ" in G.contradicted_papers("u4")


def test_critic_flags_only_graph_contradicted_paper():
    store_papers("u5", [{"id": "oa:P1", "title": "Safe Paper", "year": 2024, "venue": "",
                         "citations": 5, "authors": ["Nobody"], "abstract": "", "url": "",
                         "src": "OpenAlex", "referenced_works": [], "author_keys": ["oa:Z"]}],
                 "topic five")
    G.create_task("u5", "Researcher", "Critic", "oa:P1", status="needs_review",
                  payload={"title": "Safe Paper", "topic": "topic five", "citations": 5, "source": "OpenAlex"})
    G.store_claim_nodes("u5", "oa:P1", "Safe Paper", "topic five",
                        [{"id": "cC", "text": "We outperform every baseline on all metrics tested"}], "test")
    G.store_claim_nodes("u5", "oa:P2", "Other Paper", "topic five",
                        [{"id": "cD", "text": "Those baselines are sufficient and no improvement is needed"}], "test")
    G.store_claim_links("u5", [{"a_id": "cC", "b_id": "cD", "relation": "contradicts",
                                "evidence": "opposite"}])
    out = critic_review("u5")
    assert out["flagged"] == 1
    assert out["decided_by"] == "CONTRADICTS edges"
    assert len(G.get_tasks("u5", status="contradiction_found", to_agent="Critic")) == 1
    tasks = G.get_tasks("u5", status="needs_review", to_agent="Librarian")
    assert len(tasks) == 1
    payload = tasks[0].get("payload")
    s = str(payload)
    assert "Nobody" in s or "Librarian" in s


def test_connected_components_sizes():
    m = G._mem(G.graph_name("u6"))
    m["cites"] = [{"from": "a", "to": "b"}, {"from": "b", "to": "c"}, {"from": "x", "to": "y"}]
    comps = G.connected_components("u6")
    groups = [{mm["id"] for mm in c["members"]} for c in comps]
    assert {"a", "b", "c"} in groups          # 3-chain
    assert {"x", "y"} in groups               # pair — seed chain may add more groups


def test_mentioned_edges_use_episode_id(monkeypatch):
    calls = []
    class FakeG:
        def query(self, cypher, params=None):
            calls.append((cypher, params))
            class R:
                result_set = [[7]]
            return R()
    monkeypatch.setattr(G, "_mode", "falkordb")
    monkeypatch.setattr(G, "user_graph", lambda uid: FakeG())
    G.remember_episode("u7", "question text", mentioned_papers=["oa:AAA"],
                       mentioned_topics=["topic x"])
    assert any("id(e)" in c for c, _ in calls)
    link_calls = [(c, p) for c, p in calls if "MENTIONED" in c]
    assert len(link_calls) == 2
    assert all(p.get("eid") == 7 for _, p in link_calls)
