"""Routing regression tests.

The keyword router is the deterministic fallback and always runs in CI. The
LLM router tests need a live MISTRAL_API_KEY and are skipped without one; each
question is run 3 times with majority vote because single-pass LLM routing can
hide a 20% miss rate (the 'who knows about X' -> live_search bug was exactly
this kind of silent overlap failure).
"""
import os
import time
from collections import Counter

import pytest
from dotenv import dotenv_values

from app.agent import SPEC_TO_TOOL, route_template
from app.llm import select_tool

CASES = [
    ("Who knows about GraphRAG?", "who_knows"),
    ("Who is the expert on GraphRAG?", "who_knows"),
    ("Which authors work on graph memory?", "who_knows"),
    ("Who should I ask about FalkorDB?", "who_knows"),
    ("Anyone who has written about contradictions in RAG?", "who_knows"),
    ("who wrote graph rag papers?", "who_knows"),
    ("Latest papers on GraphRAG", "live_search"),
    ("What did I ask before?", "recall"),
    ("Which papers contradict each other?", "find_contradictions"),
    ("find papers on graph retrieval augmented generation", "live_search"),
    ("most influential paper on GraphRAG", "pagerank"),
]

_KEY = os.environ.get("MISTRAL_API_KEY") or dotenv_values(".env").get("MISTRAL_API_KEY") or ""


@pytest.mark.parametrize("q,expected", CASES)
def test_keyword_router(q, expected):
    assert route_template(q) == expected


@pytest.mark.skipif(not _KEY, reason="needs a live MISTRAL_API_KEY")
@pytest.mark.parametrize("q,expected", CASES)
def test_llm_router_majority(q, expected, monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", _KEY)

    def pick():
        """select_tool swallows 429s into a template fallback; retry a couple of
        times spaced out, and skip (not fail) if the key stays rate-limited."""
        for attempt in range(3):
            raw, _ = select_tool(q)
            if raw:
                return raw
            time.sleep(2.0 + attempt)
        return ""

    picks = []
    for _ in range(3):
        raw = pick()
        if not raw:
            pytest.skip("Mistral rate-limited (429) — routing spec unchanged")
        picks.append(SPEC_TO_TOOL.get(raw, raw))
        time.sleep(1.2)
    top = Counter(picks).most_common(1)[0][0]
    assert top == expected, f"3 runs -> {picks}"
