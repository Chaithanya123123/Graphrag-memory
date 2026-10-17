import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

from fastapi import HTTPException
from app.graph import (init_graph, remember, recall, graph_snapshot, mode, sanitize_user_id,
                       remember_episode, update_interest, store_procedure, get_procedures,
                       algorithm_engine)
from app.agent import answer

app = FastAPI(title="TraceMind - Graph Hacks (3 tracks)")


def tenant(user_id: str) -> str:
    """Every entrypoint funnels through here: rejects anything that could
    escape the user_<id> graph namespace (Cypher/graph-name injection)."""
    try:
        return sanitize_user_id(user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

class Ask(BaseModel):
    user_id: str = "chaithanya"
    question: str

class InitReq(BaseModel):
    user_id: str = "chaithanya"


class IngestReq(BaseModel):
    user_id: str = "chaithanya"
    topic: str = "graph retrieval augmented generation"
    max_results: int = 6


def write_all_memories(user_id: str, question: str, res: dict) -> dict:
    """Write all three memory kinds after each question.

    episodic  (Episode)-[:MENTIONED]->(Paper|Topic): what was asked, what it touched
    semantic  (User)-[:INTERESTED_IN {weight}]->(Topic): topic affinity that shifts
    procedural (Procedure {intent, tool_chain}): the plan that worked, reused next time
    """
    papers = [n["id"] for n in res.get("nodes", []) if n.get("type") == "Paper"][:6]
    remember(user_id, question)
    remember_episode(user_id, question, mentioned_papers=papers,
                     mentioned_topics=[res.get("topic")] if res.get("topic") else [])
    n_topics = 0
    for t in res.get("topics", [])[:3]:
        update_interest(user_id, t, delta=0.08)
        n_topics += 1
    store_procedure(user_id, intent=question[:80], tool_chain=[res.get("tool", "unknown")],
                    success=True)
    return {"episodes": 1, "interests_updated": n_topics,
            "procedures": len(get_procedures(user_id, limit=50))}

@app.get("/api/health")
def health():
    import os
    return {"ok": True, "graph_mode": mode(), "mistral": bool(os.getenv("MISTRAL_API_KEY"))}

@app.post("/api/init")
def init(r: InitReq):
    return init_graph(tenant(r.user_id))

@app.post("/api/ask")
def ask(q: Ask):
    uid = tenant(q.user_id)
    res = answer(uid, q.question)
    mem = write_all_memories(uid, q.question, res)
    stats = {"papers": len(res["nodes"]), "edges": len(res["edges"]),
             "memories": len(res["memory"]), **mem}
    return {**res, "stats": stats, "mode": mode(),
            "engines": algorithm_engine(uid)}

@app.post("/api/ingest")
def ingest(r: IngestReq):
    from app.sources import live_search, store_papers
    uid = tenant(r.user_id)
    papers = live_search(r.topic, per_src=5)
    store_papers(uid, papers, r.topic)
    return {"ingested": len(papers), "papers": [{"title": p["title"], "venue": p["venue"], "year": p["year"], "citations": p["citations"]} for p in papers]}

class AgentsReq(BaseModel):
    user_id: str = "chaithanya"
    topic: str = "graph retrieval augmented generation"

@app.post("/api/agents/run")
def run_agents(r: AgentsReq):
    """Researcher -> Critic -> Librarian handoff through Task nodes."""
    from app.agents import run_pipeline
    return run_pipeline(tenant(r.user_id), r.topic)

@app.get("/api/agents/tasks")
def agent_tasks(user_id: str = "chaithanya", status: str = None, to_agent: str = None):
    from app.graph import get_tasks
    return {"tasks": get_tasks(tenant(user_id), status=status, to_agent=to_agent)}

@app.get("/api/algorithms")
def algorithms(user_id: str = "chaithanya", topic: str = ""):
    """Graph algorithms as first-class endpoints: PageRank, communities, citation edges."""
    from app.graph import run_pagerank, community_detection, _citations
    uid = tenant(user_id)
    return {"pagerank": run_pagerank(uid, top_k=20),
            "communities": community_detection(uid),
            "citation_edges": len(_citations(uid)),
            "engines": algorithm_engine(uid)}

@app.get("/api/graph")
def graph(user_id: str = "chaithanya"): return graph_snapshot(tenant(user_id))

@app.get("/api/memory")
def memory(user_id: str = "chaithanya"): return {"memories": recall(tenant(user_id))}

# ---- serve polished frontend ----
FRONT = Path(__file__).resolve().parent.parent / "frontend"
if FRONT.exists():
    app.mount("/static", StaticFiles(directory=str(FRONT)), name="static")

@app.get("/")
def root():
    idx = FRONT / "index.html"
    if idx.exists(): return FileResponse(str(idx))
    return {"msg": "TraceMind API. Open /docs"}
