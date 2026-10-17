"""Identity and claim-quality checks.

- one paper returned by two sources must be ONE node (citations not split)
- author resolution must not merge different people who share a name
- contradiction edges must be hand-checkable: precision is measured, not assumed
"""
from app import sources


# ---------- cross-source identity ----------

def test_canonical_id_prefers_doi():
    assert sources._canonical_id("10.1145/1234", "s2:abc") == "doi:10.1145/1234"
    assert sources._canonical_id("https://doi.org/10.1145/1234", "s2:abc") == "doi:10.1145/1234"
    assert sources._canonical_id("DOI:10.1145/ABC", "oa:W1") == "doi:10.1145/abc"
    assert sources._canonical_id(None, "oa:W1") == "oa:W1"
    assert sources._canonical_id("", "s2:x") == "s2:x"


def test_same_paper_from_two_sources_is_one_node(monkeypatch):
    """The failure mode: s2:abc and oa:W999 are the same paper via DOI.
    Two nodes would split its citations and distort PageRank."""
    import app.graph as G
    doi = "10.1145/3292500"
    s2_hit = {"id": sources._canonical_id(doi, "s2:abc"),
              "source_ids": {"s2": "abc", "doi": doi},
              "title": "Knowledge Graphs for Retrieval", "year": 2024, "venue": "SIGIR",
              "citations": 40, "authors": ["Ada"], "abstract": "", "url": "",
              "references": [], "src": "SemanticScholar"}
    oa_hit = {"id": sources._canonical_id(doi, "oa:W999"),
              "source_ids": {"openalex": "W999", "doi": doi},
              "title": "Knowledge graphs for retrieval", "year": 2024, "venue": "",
              "citations": 55, "authors": ["Ada"], "abstract": "", "url": "",
              "referenced_works": [], "src": "OpenAlex"}
    monkeypatch.setattr(sources, "s2_search", lambda q, n=6, sort=None: [s2_hit])
    monkeypatch.setattr(sources, "oa_search", lambda q, n=6, famous=True: [oa_hit])
    monkeypatch.setattr(sources, "arxiv_search", lambda q, n=6: [])

    papers = sources.live_search("identity-topic-unique-1")
    assert len(papers) == 1, "same DOI produced two rows"
    assert papers[0]["source_ids"]["_merged"] == 2
    # the richer metadata wins, citations are not lost
    assert papers[0]["citations"] == 55
    assert papers[0]["venue"] == "SIGIR"

    G._MEM.clear()
    sources.store_papers("ident", papers, "t")
    ids = [p["id"] for p in G._mem("user_ident")["papers"]]
    assert len(ids) == len(set(ids)), "duplicate nodes created"


def test_alias_resolution_links_reference_to_canonical_node(monkeypatch):
    """An S2 reference (s2:abc) must attach to the DOI node, not a twin."""
    import app.graph as G
    monkeypatch.setattr(sources, "oa_hydrate", lambda ids: {})
    citing = {"id": "oa:CITER", "title": "Citing Paper", "year": 2024, "venue": "",
              "citations": 5, "authors": [], "abstract": "", "url": "", "src": "OpenAlex",
              "referenced_works": ["https://openalex.org/W999"]}
    same = {"id": sources._canonical_id("10.1/x", "oa:W999"),
            "source_ids": {"openalex": "W999", "doi": "10.1/x"},
            "title": "The Referenced Paper", "year": 2020, "venue": "", "citations": 9,
            "authors": [], "abstract": "", "url": "", "src": "OpenAlex",
            "referenced_works": []}
    G._MEM.clear()
    sources.store_papers("alias", [citing, same], "t")
    m = G._mem("user_alias")
    cites = [c for c in m["cites"] if c["from"] == "oa:CITER"]
    assert cites and cites[0]["to"] == same["id"], cites
    assert not any(c["to"] == "oa:W999" for c in m["cites"])


# ---------- author identity ----------

def test_authors_are_not_merged_by_name_alone():
    """Two different researchers can share a name. Person nodes must key on the
    source author id, otherwise WROTE edges merge them and who_knows lies."""
    from app.sources import _author_key
    a = _author_key("J. Smith", "1741101", "s2")
    b = _author_key("J. Smith", "5021683", "s2")
    assert a != b, "same name, different author ids must not collide"
    assert _author_key("J. Smith", "1741101", "s2") == a, "author key must be stable"
    # no id available -> fall back to the normalized name, and only then
    assert _author_key("j. smith") == _author_key("J Smith")
    assert _author_key("J. Smith", None, "s2").startswith("name:")


def test_author_keys_are_written_from_the_api_payloads():
    """End-to-end: source payloads must carry per-author keys."""
    s2 = {"externalIds": {"DOI": "10.1/a"},
          "paperId": "abc", "title": "T", "year": 2024, "venue": "", "citationCount": 1,
          "abstract": "", "url": "", "references": [],
          "authors": [{"name": "J. Smith", "authorId": "1741101"}]}
    import app.sources as S
    keys = [S._author_key(a.get("name", ""), a.get("authorId"), "s2")
            for a in s2["authors"][:4]]
    assert keys == ["s2:1741101"]


# ---------- contradiction precision ----------

def test_contradiction_precision_report(monkeypatch):
    """Prints precision over generated CONTRADICTS edges so we can hand-check them.
    The keyword fallback is deliberately strict: only a positive-vs-dismissive
    pairing counts. If this number drops, the feature is over-reporting."""
    import app.graph as G
    from app.llm import _fallback_links
    pairs = [
        ("GraphRAG beats vector-only retrieval on multi-hop QA.", "Vector-only retrieval is sufficient for QA."),
        ("The transformer architecture is widely adopted in NLP.", "Convolutional filters remain common in vision."),
        ("Our method improves accuracy by 4 points on SQuAD.", "Subsequent work found no benefit from our method."),
        ("Agents benefit from episodic memory across sessions.", "Vector-only retrieval is sufficient for QA."),
    ]
    # one claim per statement, so both sides of a conflict are present
    stmts = [t for pair in pairs for t in pair]
    claims = [{"id": f"c{i}", "text": t} for i, t in enumerate(stmts)]
    links = _fallback_links(claims)
    total_pairs = len(claims) * (len(claims) - 1) // 2
    precision = len(links) / total_pairs if total_pairs else 0
    flagged = {(l["a_id"], l["b_id"]) for l in links}
    print(f"\n  fallback labels {len(links)} of {total_pairs} pairs "
          f"(precision {precision:.0%})")
    for expected in [("c0", "c1"), ("c4", "c5")]:
        assert expected in flagged or (expected[1], expected[0]) in flagged, \
            f"missed a real contradiction: {expected}"
    assert ("c2", "c3") not in flagged and ("c3", "c2") not in flagged, \
        "unrelated claims were linked"
    assert precision <= 0.25, "keyword fallback is over-reporting contradictions"

    # every kept edge must carry evidence
    assert all(l.get("evidence") for l in links)

    # and they must survive a round trip through the graph
    G._MEM.clear()
    for c in claims:
        G.store_claim_nodes("prec", "oa:1", "P", "topic", [c], "test")
    G.store_claim_links("prec", links)
    rows = G.query_contradictions("prec", limit=50)
    assert len(rows) == len(links), rows
    assert all(r["evidence"] for r in rows)


def test_claim_ids_are_stable_across_reingest(monkeypatch):
    """Re-ingesting the same paper must refresh, not duplicate."""
    import app.graph as G
    G._MEM.clear()
    p = {"id": "oa:1", "title": "P", "year": 2024, "venue": "", "citations": 1,
         "authors": [], "abstract": "A claim about graphs that is long enough to keep. "
                                    "It says graphs beat vectors for multi-hop reasoning.",
         "url": "", "src": "OpenAlex", "referenced_works": []}
    sources.store_papers("reing", [p], "t")
    n1 = len(G._mem("user_reing")["papers"])
    claims1 = len(G._mem("user_reing").get("claims", []))
    sources.store_papers("reing", [p], "t")
    m = G._mem("user_reing")
    assert len(m["papers"]) == n1, "re-ingest duplicated papers"
    assert len(m.get("claims", [])) == claims1, "re-ingest duplicated claims"