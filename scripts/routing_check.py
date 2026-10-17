"""Routing regression check: LLM tool-calling + keyword template on fixed phrasings.

Each question is run N times (default 5) because LLM routing is not deterministic:
one pass hides a 20% miss rate. Reports the pick distribution and the keyword
router's answer, which is what runs when there is no API key.

    python scripts/routing_check.py 5
"""
import collections
import sys

from dotenv import load_dotenv
load_dotenv()

from app.agent import SPEC_TO_TOOL, route_template
from app.llm import select_tool

# (question, expected final tool after SPEC_TO_TOOL mapping)
CASES = [
    ("Who knows about GraphRAG?", "who_knows"),
    ("Who is the expert on GraphRAG?", "who_knows"),
    ("Which authors work on graph memory?", "who_knows"),
    ("Who should I ask about FalkorDB?", "who_knows"),
    ("Anyone who has written about contradictions in RAG?", "who_knows"),
    ("who wrote graph rag papers?", "who_knows"),
    # negative controls — must NOT over-correct into who_knows
    ("Latest papers on GraphRAG", "live_search"),
    ("What did I ask before?", "recall"),
    ("Which papers contradict each other?", "find_contradictions"),
    ("find papers on graph retrieval augmented generation", "live_search"),
    ("most influential paper on GraphRAG", "pagerank"),
]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    any_miss, flaky = 0, 0
    print(f"{n} runs per question; final tool = SPEC_TO_TOOL(llm pick)\n")
    print(f"{'question':56s} {'want':20s} {'llm picks':34s} {'template':20s}")
    print("-" * 132)
    for q, want in CASES:
        picks = [SPEC_TO_TOOL.get(p, p) or "(template)" for p, _ in (select_tool(q) for _ in range(n))]
        dist = " ".join(f"{k}:{v}" for k, v in collections.Counter(picks).most_common())
        tmpl = route_template(q)
        hit = picks.count(want)
        if hit < n:
            any_miss += 1
        if hit < max(1, 3 * n // 4):
            flaky += 1
        mark = "  OK " if hit == n else (" mix " if hit >= 3 * n // 4 else " MISS")
        print(f"{q[:54]:56s} {want:20s} {dist[:32]:34s} {tmpl:20s}{mark}")
    print("-" * 132)
    print(f"questions with any miss: {any_miss}/{len(CASES)} | FAIL-level (<75% correct): {flaky}")
    return 1 if flaky else 0


if __name__ == "__main__":
    raise SystemExit(main())
