"""Live-search behaviour with the network mocked out: dedupe, resilience, timeout cap."""
import time

from app import sources


def _p(title, cites=1, src="OpenAlex", year=2023):
    return {"id": "oa:" + title[:6].replace(" ", ""), "title": title, "year": year,
            "venue": "", "citations": cites, "authors": ["A B"], "abstract": "",
            "url": "", "src": src}


def test_live_search_dedupes_and_sorts(monkeypatch):
    monkeypatch.setattr(sources, "s2_search",
                        lambda q, n=6, sort=None: [_p("Graph RAG", 10)])
    monkeypatch.setattr(sources, "oa_search",
                        lambda q, n=6, famous=True: [_p("graph  rag!", 12)])
    monkeypatch.setattr(sources, "arxiv_search", lambda q, n=6: [])
    papers = sources.live_search("dedupe-topic-xyz")
    assert len(papers) == 1, papers


def test_one_source_down_still_works(monkeypatch):
    def boom(q, n=6, sort=None):
        raise RuntimeError("down")
    monkeypatch.setattr(sources, "s2_search", boom)
    monkeypatch.setattr(sources, "oa_search", lambda q, n=6, famous=True: [_p("OnlyOA")])
    monkeypatch.setattr(sources, "arxiv_search", lambda q, n=6: [])
    assert len(sources.live_search("resilience-topic-xyz")) == 1


def test_slow_source_is_capped(monkeypatch):
    monkeypatch.setattr(sources, "s2_search",
                        lambda q, n=6, sort=None: time.sleep(60))
    monkeypatch.setattr(sources, "oa_search", lambda q, n=6, famous=True: [])
    monkeypatch.setattr(sources, "arxiv_search", lambda q, n=6: [])
    start = time.time()
    sources.live_search("timeout-topic-xyz")
    assert time.time() - start < 20, "14s cap not enforced"


def test_store_papers_creates_real_cites(monkeypatch):
    """References must become CITES edges resolving to real nodes, not a rank chain."""
    import app.graph as G
    monkeypatch.setattr(sources, "oa_hydrate", lambda ids: {
        "oa:W999": {"id": "oa:W999", "title": "Referenced Paper 999",
                    "year": 2020, "citations": 7, "authors": [], "venue": "",
                    "abstract": "", "url": "", "src": "OpenAlex"},
        "oa:W888": {"id": "oa:W888", "title": "Referenced Paper 888",
                    "year": 2021, "citations": 3, "authors": [], "venue": "",
                    "abstract": "", "url": "", "src": "OpenAlex"},
    })
    papers = [
        {"id": "oa:AAA", "title": "Citing Paper", "year": 2024, "venue": "ACL",
         "citations": 10, "authors": ["Ada"], "abstract": "", "url": "", "src": "OpenAlex",
         "referenced_works": ["https://openalex.org/W999", "https://openalex.org/W888"]},
    ]
    G._MEM.clear()
    sources.store_papers("citestenant", papers, "t")
    m = G._mem("user_citestenant")
    cites = m.get("cites", [])
    assert len(cites) == 2, cites
    assert not any(e["from"] == e["to"] for e in cites)
    assert {e["to"] for e in cites} == {"oa:W999", "oa:W888"}
    # referenced papers become nodes carrying their real hydrated titles
    by_id = {p["id"]: p for p in m["papers"]}
    assert by_id["oa:W999"]["title"] == "Referenced Paper 999"
    assert by_id["oa:W888"]["citations"] == 3


def test_unhydrated_reference_still_gets_a_node():
    """A reference OpenAlex cannot resolve must not vanish from the graph."""
    import app.graph as G
    G._MEM.clear()
    sources.store_papers("stubtenant", [{
        "id": "oa:CCC", "title": "Citing", "year": 2024, "venue": "", "citations": 1,
        "authors": [], "abstract": "", "url": "", "src": "OpenAlex",
        "referenced_works": ["https://openalex.org/W404"]}], "t")
    ids = {p["id"] for p in G._mem("user_stubtenant")["papers"]}
    assert "oa:W404" in ids


def test_store_papers_is_idempotent():
    import app.graph as G
    papers = [{"id": "oa:AAA", "title": "Citing", "year": 2024, "venue": "",
               "citations": 1, "authors": [], "abstract": "", "url": "", "src": "OpenAlex",
               "referenced_works": ["https://openalex.org/W999"]}]
    G._MEM.clear()
    sources.store_papers("idem", papers, "t")
    sources.store_papers("idem", papers, "t")
    m = G._mem("user_idem")
    assert len([c for c in m["cites"] if c["to"] == "oa:W999"]) == 1