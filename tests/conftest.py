"""Shared test setup: no network, no LLM, fresh in-memory graphs.

Without this, every ingest fires real Semantic Scholar/OpenAlex requests and
Mistral claim-extraction calls, which makes the suite slow and flaky. The
algorithmic behaviour under test is unchanged.
"""
import os

import pytest

import app.graph as G
from app import llm, sources


@pytest.fixture(autouse=True)
def force_memory(monkeypatch):
    """Deterministic suite: tests key data in G._MEM, so run in memory mode
    whether Docker is up or not. Opt into the real backend with
    TRACEMIND_TEST_FALKOR=1."""
    if os.getenv("TRACEMIND_TEST_FALKOR") != "1":
        monkeypatch.setattr(G, "_mode", "memory")
    yield


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(sources, "oa_hydrate", lambda ids: {})
    yield


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    """Force the deterministic fallbacks so no Mistral key is needed."""
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.setattr(llm, "select_tool", lambda q: ("", "template"))
    yield


@pytest.fixture(autouse=True)
def clean_graphs():
    """Each test starts with empty tenant graphs."""
    G._MEM.clear()
    yield
    G._MEM.clear()


@pytest.fixture
def papers():
    return [
        {"id": "oa:AAA", "title": "Graph RAG Survey", "year": 2024, "venue": "ACM",
         "citations": 120, "authors": ["Ada Lovelace"],
         "abstract": "Graph retrieval beats vector-only retrieval on multi-hop "
                     "question answering, and this is demonstrated at scale.",
         "url": "", "src": "OpenAlex", "referenced_works": []},
        {"id": "oa:BBB", "title": "Vector-Only Retrieval Limits", "year": 2023,
         "venue": "ACL", "citations": 90, "authors": ["Alan Turing"],
         "abstract": "Vector-only retrieval is sufficient for question answering "
                     "and additional structure is unnecessary.",
         "url": "", "src": "OpenAlex", "referenced_works": []},
    ]