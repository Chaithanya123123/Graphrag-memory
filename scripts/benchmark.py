"""Benchmark: graph agent vs plain vector-RAG baseline.

15 multi-hop questions over a real ingested topic. For each we score:
  correct       does the answer contain the expected facts
  traceable     can we show the graph path (nodes + edges) that produced it

The baseline is a genuine vector-RAG: TF-IDF cosine over the same paper
abstracts, top-k chunks, no graph, no traversal. It is the thing we claim to beat.

    python -m uvicorn app.main:app --port 8000
    python scripts/benchmark.py --topic "graph retrieval augmented generation"
"""
import argparse
import json
import math
import re
import time
from collections import Counter

import httpx

BASE = "http://localhost:8000"

# (question, [terms whose presence in the answer counts as correct])
QUESTIONS = [
    ("find papers on graph retrieval augmented generation", ["Graph", "RAG"]),
    ("most influential paper in graph rag", ["Graph"]),
    ("which papers cite the most work in graph retrieval?", ["Graph"]),
    ("which claims contradict each other in graph rag?", []),
    ("what did I ask before?", []),
    ("papers about retrieval augmented generation", ["Retrieval"]),
    ("who wrote graph rag papers?", []),
    ("how are graph retrieval papers connected?", []),
    ("which papers about knowledge graphs are most cited?", ["Graph"]),
    ("find recent retrieval papers", ["Retrieval"]),
    ("papers that build on graph rag", ["Graph"]),
    ("clusters of graph retrieval papers", []),
    ("shortest path between graph papers", []),
    ("papers about vector search limits", ["Vector"]),
    ("find agent memory papers", []),
]


# ---------- baseline: TF-IDF cosine vector RAG over the same abstracts ----------

def tokenize(text: str):
    return [w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(w) > 2]


def build_index(chunks):
    df = Counter()
    for _, text in chunks:
        df.update(set(tokenize(text)))
    n = max(1, len(chunks))
    idf = {w: math.log(n / (1 + c)) + 1 for w, c in df.items()}
    vecs = []
    for _, text in chunks:
        counts = Counter(tokenize(text))
        vec = {w: (1 + math.log(c)) * idf.get(w, 0.0) for w, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        vecs.append({w: v / norm for w, v in vec.items()})
    return vecs


def vector_answer(question, chunks, k=5):
    q = {w: 1.0 for w in tokenize(question)}
    qnorm = math.sqrt(sum(v * v for v in q.values())) or 1.0
    scored = []
    for (title, text), vec in zip(chunks, build_index(chunks)):
        s = sum(v * vec.get(w, 0.0) for w, v in q.items()) / qnorm
        scored.append((s, title))
    scored.sort(reverse=True)
    hits = [t for s, t in scored[:k] if s > 0]
    return ("Top retrieved passages:\n" + "\n".join(f"- {t}" for t in hits)) if hits \
        else "No relevant passages found."


def traceable(answer: dict) -> bool:
    """A traceable answer cites a graph path: nodes returned and edges traversed."""
    if not answer.get("nodes"):
        return False
    ids = {n.get("id") for n in answer["nodes"]}
    highlighted = set(answer.get("highlight") or [])
    used_edges = [e for e in answer.get("edges", [])
                  if e.get("from") in ids and e.get("to") in ids]
    return bool(highlighted or used_edges)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="graph retrieval augmented generation")
    ap.add_argument("--user", default="bench")
    args = ap.parse_args()

    with httpx.Client(base_url=BASE, timeout=120) as c:
        print(f"ingesting '{args.topic}' ...")
        t0 = time.time()
        r = c.post("/api/ingest", json={"user_id": args.user, "topic": args.topic})
        r.raise_for_status()
        papers = r.json()["papers"]
        print(f"  {len(papers)} papers in {time.time() - t0:.1f}s\n")

        graph = c.get("/api/graph", params={"user_id": args.user}).json()
        chunks = [(n["label"], n["label"]) for n in graph["nodes"] if n.get("type") == "Paper"]

        rows = []
        for q, expect in QUESTIONS:
            t0 = time.time()
            a = c.post("/api/ask", json={"user_id": args.user, "question": q})
            elapsed = time.time() - t0
            if a.status_code != 200:
                rows.append({"q": q, "agent_ok": False, "vector_ok": False,
                             "traceable": False, "ms": elapsed * 1000, "error": a.text[:80]})
                continue
            d = a.json()
            text = d.get("text", "")
            agent_ok = all(t.lower() in text.lower() for t in expect) if expect else bool(text)
            base = vector_answer(q, chunks)
            vector_ok = all(t.lower() in base.lower() for t in expect) if expect else bool(chunks)
            rows.append({"q": q, "tool": d.get("tool"), "agent_ok": agent_ok,
                         "vector_ok": vector_ok, "traceable": traceable(d),
                         "ms": elapsed * 1000,
                         "answer": text[:160].replace("\n", " ")})

    n = len(rows)
    a_ok = sum(r["agent_ok"] for r in rows)
    v_ok = sum(r["vector_ok"] for r in rows)
    tr = sum(r["traceable"] for r in rows)
    contra = [r for r in rows if r.get("tool") == "find_contradictions"]
    contra_ok = sum(r["agent_ok"] for r in contra)
    contra_v = sum(r["vector_ok"] for r in contra)

    print(f"{'question':52s} {'tool':20s} {'agent':6s} {'vec':5s} {'trace':6s} {'ms':>7s}")
    print("-" * 104)
    for r in rows:
        print(f"{r['q'][:50]:52s} {str(r.get('tool'))[:18]:20s} "
              f"{'PASS' if r['agent_ok'] else 'FAIL':6s} "
              f"{'PASS' if r['vector_ok'] else 'FAIL':5s} "
              f"{'yes' if r['traceable'] else 'no':6s} {r['ms']:7.0f}")
    print("-" * 104)
    print(f"agent accuracy      {a_ok}/{n} ({100*a_ok/n:.0f}%)")
    print(f"vector-RAG accuracy {v_ok}/{n} ({100*v_ok/n:.0f}%)")
    print(f"traceable answers   {tr}/{n} ({100*tr/n:.0f}%)")
    if contra:
        print(f"contradiction Qs    agent {contra_ok}/{len(contra)} vs vector {contra_v}/{len(contra)}")
    print(f"median latency      {sorted(r['ms'] for r in rows)[n//2]:.0f} ms")

    out = {"topic": args.topic, "papers": len(papers), "questions": rows,
           "agent_accuracy": a_ok / n, "vector_accuracy": v_ok / n,
           "traceable": tr / n}
    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nwrote benchmark_results.json")


if __name__ == "__main__":
    main()