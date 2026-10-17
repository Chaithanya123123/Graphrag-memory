"""API-level tests: routing, grounding, memory, tenant isolation, injection.

These run without network and without FalkorDB (the in-memory fallback is a
supported backend), so they pass on a clean clone.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def offline(monkeypatch, papers):
    """Never touch the network in tests."""
    from app import sources
    monkeypatch.setattr(sources, "live_search", lambda q, per_src=6: list(papers))
    yield


def ask(user: str, q: str):
    r = client.post("/api/ask", json={"user_id": user, "question": q})
    assert r.status_code == 200, r.text
    return r.json()


def setup_module():
    for u in ("testa", "testb"):
        client.post("/api/init", json={"user_id": u})


def test_response_shape():
    d = ask("testa", "Who knows GraphRAG?")
    for key in ("text", "tool", "cypher", "nodes", "edges", "memory"):
        assert key in d, f"missing {key}"
    assert d["nodes"] and d["edges"]


def test_bad_user_id_rejected():
    for bad in ("x; DROP", "a' DETACH DELETE n", "A_b", "z" * 40, "has-hyphen"):
        r = client.post("/api/ask", json={"user_id": bad, "question": "hi"})
        assert r.status_code in (400, 422), f"{bad!r} was accepted"


BAD_IDS = ["x; DROP", "a' DETACH DELETE n", "A_b", "z" * 40, "has-hyphen",
           "a' CONTRADICTS) DETACH DELETE (n // `"]


@pytest.mark.parametrize("method,path,body,query", [
    ("POST", "/api/ask", {"user_id": "{bad}", "question": "hi"}, None),
    ("POST", "/api/init", {"user_id": "{bad}"}, None),
    ("POST", "/api/ingest", {"user_id": "{bad}", "topic": "t"}, None),
    ("POST", "/api/agents/run", {"user_id": "{bad}", "topic": "t"}, None),
    ("GET", "/api/graph", None, {"user_id": "{bad}"}),
    ("GET", "/api/memory", None, {"user_id": "{bad}"}),
    ("GET", "/api/algorithms", None, {"user_id": "{bad}"}),
    ("GET", "/api/agents/tasks", None, {"user_id": "{bad}"}),
])
def test_every_endpoint_rejects_bad_tenant(method, path, body, query):
    """Every entry point must gate the tenant, not just /api/ask."""
    for bad in BAD_IDS:
        if method == "POST":
            r = client.post(path, json={k: v.format(bad=bad) if isinstance(v, str) else v
                                        for k, v in body.items()})
        else:
            r = client.get(path, params={k: v.format(bad=bad) for k, v in query.items()})
        assert r.status_code in (400, 422), f"{path} accepted {bad!r} -> {r.status_code}"


@pytest.mark.parametrize("tool,args", [
    ("pagerank", {}),
    ("find_contradictions", {}),
    ("shortest_path", {"paper_a": "oa:a", "paper_b": "oa:b"}),
    ("community_detection", {}),
    ("who_knows", {"method": "GraphRAG"}),
    ("get_user_interests", {}),
    ("get_procedures", {}),
    ("run_agents", {"topic": "t"}),
    ("create_handoff_task", {"from_agent": "R", "to_agent": "C", "paper_id": "x"}),
    ("get_agent_tasks", {"agent": "Critic"}),
])
def test_mcp_every_tool_rejects_bad_tenant(tool, args):
    """The MCP route is the easiest one to forget; it must reject too."""
    from app.mcp_server import app as mcp_app
    m = TestClient(mcp_app)
    for bad in BAD_IDS:
        r = m.post("/mcp/call", json={"name": tool,
                                       "arguments": {**args, "user_id": bad}})
        assert r.status_code == 400, f"mcp {tool} accepted {bad!r} -> {r.status_code}"


def test_relation_name_never_from_request():
    """CONTRADICTS/SUPPORTS are chosen from a whitelist, never interpolated from
    user input. A relation-looking string must not reach the query builder."""
    from app.graph import store_claim_links
    payload = {"a_id": "c1", "b_id": "c2",
               "relation": "CONTRADICTS) DETACH DELETE (n // `",
               "evidence": "x"}
    store_claim_links("whitelistuser", [payload])
    d = client.get("/api/graph", params={"user_id": "whitelistuser"}).json()
    labels = {e["label"] for e in d["edges"]}
    # the edge is rendered with a whitelisted relation, never the injected string
    claim_edges = {e["label"] for e in d["edges"]
                   if e["from"] == "c1" or e["to"] == "c2"}
    assert claim_edges <= {"CONTRADICTS", "SUPPORTS", "CLAIMS", "ABOUT"}, claim_edges
    assert not any("DETACH" in l for l in labels), labels


def test_quote_in_input_does_not_break():
    d = ask("testa", "what is O'Reilly's \"graph\" approach for 'retrieval'?")
    assert d["text"]
    assert "tool" in d


def test_live_search_answer_is_shaped_by_traversal():
    d = ask("testa", "find papers on graph rag")
    assert d["tool"] == "live_search"
    # the answer must come from an algorithm/traversal, not just a list
    assert any(k in d["text"].lower() for k in ("pagerank", "traversal", "community", "influence"))
    assert d["edges"], "ingest produced no edges"


def test_memory_persists():
    ask("testa", "unique-marker-question-12345")
    d = ask("testa", "what did I ask before?")
    assert "unique-marker-question-12345" in str(d)


def test_tenant_isolation():
    ask("testa", "secret-topic-from-a")
    d = ask("testb", "what did I ask before?")
    assert "secret-topic-from-a" not in str(d)
    g = client.get("/api/graph", params={"user_id": "testb"}).json()
    assert "secret-topic-from-a" not in str(g)


def test_three_memory_types_written():
    d = ask("testa", "tell me about graph rag papers")
    st = d["stats"]
    assert st["episodes"] >= 1          # episodic
    assert st["interests_updated"] >= 1  # semantic
    assert st["procedures"] >= 1        # procedural


def test_edges_carry_provenance():
    ask("testa", "find papers on graph rag")
    g = client.get("/api/graph", params={"user_id": "testa"}).json()
    assert g["edges"]
    assert all("source" in e for e in g["edges"])
    assert all("retrieved_at" in e for e in g["edges"])


def test_algorithms_endpoint():
    ask("testa", "find papers on graph rag")
    d = client.get("/api/algorithms", params={"user_id": "testa"}).json()
    assert {"pagerank", "communities", "citation_edges", "engines"} <= set(d)
    # the demo must state which implementation actually ran
    assert d["engines"]["pagerank"] in ("falkordb", "python")
    assert d["engines"]["backend"] in ("falkordb", "memory")


def test_agent_pipeline_runs():
    r = client.post("/api/agents/run", json={"user_id": "testa", "topic": "graph rag"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"researcher", "critic", "librarian"}
    assert body["researcher"]["papers"] >= 1