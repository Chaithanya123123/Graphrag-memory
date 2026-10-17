"""Mistral grounding: polish graph facts into a natural answer. Never invents — cites the path."""
import os
import httpx

MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
API = "https://api.mistral.ai/v1/chat/completions"

SYSTEM = ("You are TraceMind, a graph-native research agent. You answer ONLY from the "
          "FalkorDB graph facts given. Rules: 1) Every claim must trace to a node/edge in the facts. "
          "2) Name the traversal (e.g. CITES*1..4, CONTRADICTS) and the Cypher used. "
          "3) If facts are thin, say so — never hallucinate papers or people. "
          "4) End with a one-line 'Graph path:' summary. Keep answers under 180 words.")

def takeaway(question: str, titles: list) -> tuple[str, str]:
    """2-line takeaway only — never the list itself (list stays deterministic)."""
    key = os.getenv("MISTRAL_API_KEY", "")
    if not key or not titles:
        return "", "template"
    try:
        r = httpx.post(API, headers={"Authorization": f"Bearer {key}"},
                       json={"model": MODEL, "messages": [
                           {"role": "system", "content": "In 2 sentences max, note the trend across these paper titles (venues/years). No new facts, no paper names not listed."},
                           {"role": "user", "content": f"Question: {question}\nTitles:\n" + "\n".join(titles[:12])}],
                             "temperature": 0.2, "max_tokens": 150},
                       timeout=25)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"], f"mistral:{MODEL}"
    except Exception as e:
        print("mistral fallback:", e)
        return "", "template"

def polish(question: str, tool: str, cypher: str, base_text: str, memory: list) -> tuple[str, str]:
    key = os.getenv("MISTRAL_API_KEY", "")
    if not key:
        return base_text, "template"
    mem_txt = "\n".join(memory[-5:]) if memory else "(no prior memory)"
    user = (f"Question: {question}\nTool used: {tool}\nCypher: {cypher}\n"
            f"Graph facts (authoritative):\n{base_text}\nPast memories:\n{mem_txt}\n"
            f"Write the final grounded answer.")
    try:
        r = httpx.post(API, headers={"Authorization": f"Bearer {key}"},
                       json={"model": MODEL, "messages": [
                           {"role": "system", "content": SYSTEM},
                           {"role": "user", "content": user}],
                             "temperature": 0.2, "max_tokens": 500},
                       timeout=25)
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"]
        return txt, f"mistral:{MODEL}"
    except Exception as e:
        print("mistral fallback:", e)
        return base_text, "template"


# ---- real claims + contradiction detection (Track 1 differentiator) ----
import json as _json
import re as _re


def _fallback_claims(title: str, abstract: str) -> list:
    """No-key fallback: split abstract into 1-2 sentence claims, no hallucination."""
    if not abstract:
        return []
    sents = _re.split(r"(?<=[.!?])\s+", abstract.strip())
    sents = [s.strip()[:200] for s in sents if len(s.strip()) > 40][:2]
    return sents


def extract_claims(title: str, abstract: str, max_claims: int = 3) -> tuple[list, str]:
    """Extract 1-3 atomic claims from title+abstract. Returns (claims, writer)."""
    if not abstract:
        return [], "template"
    key = os.getenv("MISTRAL_API_KEY", "")
    if not key:
        return _fallback_claims(title, abstract), "template"
    prompt = (
        f"Paper title: {title}\nAbstract: {abstract[:1200]}\n\n"
        f"Extract up to {max_claims} atomic factual claims made by THIS paper. "
        "Each claim = one sentence, <30 words, no citations, no new facts. "
        'Return JSON list only: [{"text": "..."}]'
    )
    try:
        r = httpx.post(API, headers={"Authorization": f"Bearer {key}"},
                       json={"model": MODEL, "messages": [
                           {"role": "system", "content": "You extract only claims stated in the given abstract. Never invent. JSON list only."},
                           {"role": "user", "content": prompt}],
                             "temperature": 0.1, "max_tokens": 300},
                       timeout=25)
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"]
        m = _re.search(r"\[.*\]", txt, _re.DOTALL)
        data = _json.loads(m.group(0) if m else txt)
        out = [d.get("text", "").strip()[:250] for d in data if isinstance(d, dict) and d.get("text")]
        return [c for c in out if len(c) > 10][:max_claims], f"mistral:{MODEL}"
    except Exception as e:
        print("extract_claims fallback:", e)
        return _fallback_claims(title, abstract), "template"


def find_claim_links(claims: list) -> tuple[list, str]:
    """Label pairs as supports/contradicts/neutral. Keep only non-neutral with evidence.

    claims: [{id, text, paper}]. Batched in ONE LLM call to avoid O(n^2) calls.
    Returns ([{a_id, b_id, relation, evidence}], writer).
    """
    # dedupe + cap to bound cost: max 15 claims -> 1 LLM call
    uniq, seen = [], set()
    for c in claims:
        t = (c.get("text") or "").strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            uniq.append(c)
    uniq = uniq[:15]
    if len(uniq) < 2:
        return [], "template"
    key = os.getenv("MISTRAL_API_KEY", "")
    if not key:
        return _fallback_links(uniq), "template"
    listing = "\n".join(f'{c["id"]}: {c["text"]} (paper: {c.get("paper","")[:60]})' for c in uniq)
    prompt = (
        "Claims from different papers on the same topic:\n" + listing +
        "\n\nFind pairs where one claim SUPPORTS or CONTRADICTS another. "
        "Ignore neutral/unrelated pairs. For each kept pair quote the conflicting phrase as evidence. "
        'Return JSON list only: [{"a_id":"...","b_id":"...","relation":"supports|contradicts","evidence":"..."}]'
    )
    try:
        r = httpx.post(API, headers={"Authorization": f"Bearer {key}"},
                       json={"model": MODEL, "messages": [
                           {"role": "system", "content": "You label only supports/contradicts with textual evidence. JSON list only."},
                           {"role": "user", "content": prompt}],
                             "temperature": 0.1, "max_tokens": 600},
                       timeout=30)
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"]
        m = _re.search(r"\[.*\]", txt, _re.DOTALL)
        data = _json.loads(m.group(0) if m else txt)
        valid_ids = {c["id"] for c in uniq}
        out = []
        for d in data:
            if not isinstance(d, dict):
                continue
            rel = (d.get("relation") or "").lower()
            if rel not in ("supports", "contradicts"):
                continue
            if d.get("a_id") not in valid_ids or d.get("b_id") not in valid_ids:
                continue
            if d.get("a_id") == d.get("b_id"):
                continue
            out.append({"a_id": d["a_id"], "b_id": d["b_id"], "relation": rel,
                        "evidence": (d.get("evidence") or "")[:250]})
        return out[:30], f"mistral:{MODEL}"
    except Exception as e:
        print("find_claim_links fallback:", e)
        return _fallback_links(uniq), "template"


def _fallback_links(claims: list) -> list:
    """Keyword heuristic when no LLM key: beats/outperforms vs sufficient/no-benefit."""
    pos = ["beats", "outperform", "better than", "improves", "state of the art", "sota"]
    neg = ["sufficient", "no benefit", "not needed", "fails", "limited", "unnecessary", "on par"]
    out = []
    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            a, b = claims[i]["text"].lower(), claims[j]["text"].lower()
            a_pos, b_pos = any(k in a for k in pos), any(k in b for k in pos)
            a_neg, b_neg = any(k in a for k in neg), any(k in b for k in neg)
            if (a_pos and b_neg) or (a_neg and b_pos):
                out.append({"a_id": claims[i]["id"], "b_id": claims[j]["id"],
                            "relation": "contradicts", "evidence": "keyword heuristic: positive vs dismissive phrasing"})
    return out


# ---- tool selection by description (tool calling), template router as fallback ----

TOOL_SPECS = [
    {"name": "find_contradictions",
     "description": "Find claims from different papers that disagree, refute or contradict each other, with the evidence sentence. "
                    "Use for 'what contradicts X', 'which claims disagree'. Not for finding papers or people."},
    {"name": "find_supports",
     "description": "Find claims that corroborate or support each other across papers. Not for finding papers or people."},
    {"name": "influential_papers",
     "description": "Most influential / highly cited / key papers on a topic, ranked. Uses PageRank over citation edges. "
                    "Returns papers, not people."},
    {"name": "shortest_path",
     "description": "How two specific papers are connected, citation path, lineage, who cites whom between two papers."},
    {"name": "communities",
     "description": "Cluster related papers into groups, communities, clusters or subtopics."},
    {"name": "who_knows",
     "description": "Use when the question asks WHICH PEOPLE or authors know, wrote about, work on or are experts on a topic. "
                    "Triggers: 'who knows', 'who is the expert', 'who wrote about', 'who works on', 'who should I ask', 'anyone who has written'. "
                    "Returns people from the graph, not papers. Even if the sentence ends with 'papers', a 'who' subject means this tool."},
    {"name": "recall",
     "description": "What the user asked before, previous questions, conversation history, memory."},
    {"name": "live_search",
     "description": "Use ONLY when the user asks to find, fetch or update papers on a topic from the web "
                    "('find papers on...', 'latest papers', 'recent work', 'surveys'). "
                    "Never use for questions about people, memory, or relationships already in the graph."},
]


def select_tool(question: str) -> tuple[str, str]:
    """Let the LLM pick a tool by reading the tool descriptions (tool calling).

    Falls back to the keyword template router when there is no API key or the call
    fails, so routing never breaks the demo."""
    key = os.getenv("MISTRAL_API_KEY", "")
    if not key:
        return "", "template"
    catalog = "\n".join(f'- {t["name"]}: {t["description"]}' for t in TOOL_SPECS)
    prompt = (f"Tools:\n{catalog}\n\nQuestion: {question}\n\n"
              "Choose the single best tool by its description. Reply with the tool name only.")
    try:
        r = httpx.post(API, headers={"Authorization": f"Bearer {key}"},
                       json={"model": MODEL, "messages": [
                           {"role": "system", "content": "You are a tool router. Reply with exactly one tool name, nothing else."},
                           {"role": "user", "content": prompt}],
                           "temperature": 0.0, "max_tokens": 20},
                       timeout=15)
        r.raise_for_status()
        name = r.json()["choices"][0]["message"]["content"].strip().strip(".").split()[0]
        valid = {t["name"] for t in TOOL_SPECS}
        if name in valid:
            return name, f"mistral:{MODEL}"
        print("router returned unknown tool:", name)
        return "", f"mistral:{MODEL}"
    except Exception as e:
        print("router fallback:", e)
        return "", "template"
