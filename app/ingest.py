"""arXiv ingestion: public data -> FalkorDB graph (Track 1 + 3)."""
import arxiv

def fetch(topic: str, max_results: int = 8):
    client = arxiv.Client()
    search = arxiv.Search(query=topic, max_results=max_results,
                          sort_by=arxiv.SortCriterion.SubmittedDate)
    out = []
    for r in client.results(search):
        out.append({"id": r.entry_id.split("/")[-1], "title": r.title.strip(),
                    "authors": [a.name for a in r.authors[:4]], "year": r.published.year,
                    "summary": r.summary[:600]})
    return out

def ingest_to_graph(user_id: str, topic: str, max_results: int = 8):
    from app.graph import user_graph, mode, _mem, sanitize_user_id
    papers = fetch(topic, max_results)
    if mode() == "falkordb":
        g = user_graph(user_id)
        for p in papers:
            g.query("MERGE (p:Paper {id:$id}) SET p.title=$t, p.year=$y",
                    params={"id": p["id"], "t": p["title"][:120], "y": p["year"]})
            g.query("MERGE (m:Method {name:$topic}) WITH m MATCH (p:Paper {id:$id}) MERGE (p)-[:USES_METHOD]->(m)",
                    params={"topic": topic, "id": p["id"]})
            for au in p["authors"]:
                g.query("MERGE (a:Person {name:$n}) WITH a MATCH (p:Paper {id:$id}) MERGE (a)-[:WROTE]->(p)",
                        params={"n": au, "id": p["id"]})
        # Note: arXiv doesn't provide citation data directly; CITES edges will come from Semantic Scholar/OpenAlex via live_search
    else:
        m = _mem(f"user_{sanitize_user_id(user_id)}")
        for p in papers:
            if p["id"] not in [x["id"] for x in m["papers"]]:
                m["papers"].append({"id": p["id"], "title": p["title"][:80], "year": p["year"]})
    return papers
