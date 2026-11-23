# TraceMind — Graph-Native Research Agent

[![CI](https://github.com/Chaithanya123123/Graphrag-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/Chaithanya123123/Graphrag-memory/actions/workflows/ci.yml)

TraceMind is a graph-native research assistant built for Graph Hacks with FalkorDB.
Instead of vector-only RAG, every answer is produced by a real graph traversal
(CITES, CONTRADICTS, WROTE, USES_METHOD, REMEMBERS) and the Cypher behind it is
shown in the UI as explainability proof.

It covers 3 tracks:
- **Track 1 · Connected data:** Papers, Authors, Venues, Topics linked in FalkorDB. Live ingest from Semantic Scholar + OpenAlex + arXiv.
- **Track 2 · Memory:** `(User)-[:REMEMBERS]->(Episode)` persists across sessions per `user_id` (tenant-isolated graph `user_<id>`).
- **Track 3 · Lab brain:** Claims, Contradictions, Experts, Lineage, Projects answered from the graph, polished by Mistral (with template fallback).

## Architecture

```
User Question (frontend/index.html)
        |
        v
FastAPI app/main.py -> /api/ask
        |
        +-> app/agent.py : route() picks 1 tool
        |     live_search | trace_lineage | find_contradictions | who_knows | recall
        |
        +-> app/sources.py : live_search() -> S2 + OpenAlex + arXiv in parallel threads, dedupe, sort by citations+year
        +-> app/graph.py : store_papers(), remember(), recall(), graph_snapshot() -> FalkorDB or memory fallback
        +-> app/llm.py : takeaway() + polish() via Mistral API, fallback to template if no key
        |
        v
Return { text, tool, cypher, nodes, edges, highlight, memory, stats, mode }
        |
        v
frontend/app.js renders markdown answer + Cytoscape.js graph + Cypher panel
```

Backend serves frontend itself: `app/main.py` mounts `frontend/` at `/static` and `/` returns `index.html`.

## Tech Stack

- Backend: FastAPI + Uvicorn + Pydantic + python-dotenv
- Graph: FalkorDB (`falkordb/falkordb:v4.22.1`, pinned in docker-compose.yml; ports 6379 + 3000) with in-memory fallback if Docker is off
- Live data: Semantic Scholar API + OpenAlex API + `arxiv` package via `httpx`
- LLM: Mistral `chat/completions` (`MISTRAL_MODEL=open-mistral-7b` by default), template fallback
- Frontend: Tailwind CDN + Cytoscape.js 3.28.1 + Marked, no build step

## Project Structure

```
tracemind/
  app/
    main.py      # FastAPI routes: /api/health, /api/init, /api/ask, /api/ingest, /api/graph, /api/memory + serve frontend
    agent.py     # router + answer(): question -> tool -> graph query -> formatted text
    graph.py     # FalkorDB connect, init_graph(), remember(), recall(), graph_snapshot(), SEED data
    sources.py   # s2_search(), oa_search(), arxiv_search(), live_search(), store_papers()
    llm.py       # Mistral polish() + takeaway(), never invents, cites path
    ingest.py    # legacy arXiv-only fetch() + ingest_to_graph()
  frontend/
    index.html   # layout: sidebar (user, ingest, samples, stats) + chat + live graph + cypher panel
    app.js       # api(), ask(), draw(), refreshGraph(), sample prompts
    styles.css
  docker-compose.yml # falkordb service
  requirements.txt   # falkordb, fastapi, uvicorn, pydantic, python-dotenv, arxiv, httpx
  .env / .env.example # MISTRAL_API_KEY, MISTRAL_MODEL, FALKOR_HOST, FALKOR_PORT
```

Key APIs (`app/main.py`):

- `GET /api/health` -> `{ok, graph_mode, mistral}`
- `POST /api/init?user_id=chaithanya` -> seeds Paper/Method/Claim/Person/Project + CITES/CONTRADICTS edges
- `POST /api/ask {user_id, question}` -> runs agent, writes episodic memory, returns text+cypher+viz
- `POST /api/ingest {user_id, topic, max_results}` -> live_search + store_papers
- `GET /api/graph?user_id=` -> `{nodes, edges}` for Cytoscape
- `GET /api/memory?user_id=` -> `{memories}`

Router logic (`app/agent.py:19`):

- contains `remember|before|last time|history|my memory` -> `recall`
- contains `contradict|disagree|counter|refute` -> `find_contradictions`
- contains `who knows|who worked|expert|owner` -> `who_knows`
- contains `lineage|builds on|shortest path` -> `trace_lineage`
- else -> `live_search` (recent + most-cited, stores into your graph)

## Prerequisites

1. Python 3.10+ (`python --version`)
2. Docker Desktop running (for FalkorDB - optional, app falls back to `memory` mode)
3. Internet (for S2/OpenAlex/arXiv + Mistral)
4. Mistral API key (optional - works in `template` mode without it)

## Steps to Run - Detailed

### 1. Clone the project

```powershell
git clone https://github.com/Chaithanya123123/Graphrag-memory.git
cd Graphrag-memory
```

### 2. Configure environment variables

Create `.env` from the example:

```powershell
Copy-Item ".env.example" ".env"
```

Edit `.env`:

```
MISTRAL_API_KEY=mstrl_...      # optional, without it llm=template
MISTRAL_MODEL=open-mistral-7b
FALKOR_HOST=localhost
FALKOR_PORT=6379
```

Explanation: `FALKOR_HOST/PORT` tells `app/graph.py` where FalkorDB lives. `MISTRAL_*` tells `app/llm.py` how to polish answers.

### 3. Start FalkorDB with Docker

```powershell
docker compose up -d
docker ps
```

What this does: pulls `falkordb/falkordb:v4.22.1`, starts it on `localhost:6379` (protocol) and `localhost:3000` (browser UI), creates volume `falkordb-data`.

Verify: open `http://localhost:3000` or check health later shows `"graph_mode":"falkordb"`. If Docker is off, app logs `[graph] FalkorDB unavailable (...) -> DEMO memory mode` and still runs.

To stop later: `docker compose down`

### 4. Install Python dependencies

```powershell
pip install -r requirements.txt
```

Installs: `falkordb, fastapi, uvicorn, pydantic, python-dotenv, arxiv, httpx`.

If `pip` is blocked, try `python -m pip install -r requirements.txt`.

### 5. Start backend

```powershell
uvicorn app.main:app --reload --port 8000
```

What this does: loads `.env`, connects to FalkorDB, mounts `frontend/` at `/static`, serves API on port 8000. `--reload` auto-restarts on code change.

Alternative: `python -m uvicorn app.main:app --reload --port 8000`

Leave this terminal running. You should see `[graph] FalkorDB connected at localhost:6379` or fallback message.

### 6. Open in browser

- App UI: `http://localhost:8000`
- API docs (Swagger): `http://localhost:8000/docs`
- Health check: `http://localhost:8000/api/health` should return `{"ok":true,"graph_mode":"falkordb","mistral":true}`

If UI shows `mode: offline — start backend`, backend is not running or wrong port.

### 7. Use the app (happy path demo for judges)

1. Keep `USER` as `chaithanya` (each user gets isolated graph `user_chaithanya`).
2. Click `Init graph` -> creates seed: `Graph RAG Survey, FalkorDB Memory Paper, Vector-Only Limits + GraphRAG + Claims c1/c2 + Chaithanya + TraceMind project`.
3. Click sample or type in chat and press `Ask`:
   - `Famous CVPR papers on diffusion models` -> live_search demo (recent + cited)
   - `Latest IEEE work on vision transformers` -> live_search
   - `What contradicts the claim that graph retrieval beats vector-only?` -> `(c1)-[:CONTRADICTS]->(c2)` traversal
   - `Who knows GraphRAG in our lab?` -> `(Person)-[:WROTE]->(Paper)-[:USES_METHOD]->(Method)`
   - `trace lineage of Graph RAG Survey` -> `CITES*1..4` path
   - `what did I ask before?` -> `(User)-[:REMEMBERS]->(Episode)` persistence proof
4. Check right panel: `LIVE GRAPH` visualizes nodes/edges returned by that exact query. Expand `Cypher behind last answer` to show judges the explainability.
5. `INGEST PUBLIC PAPERS`: enter topic e.g. `graph retrieval augmented generation` -> `Add` -> fetches arXiv and links `Paper-USES_METHOD->Method`, `Person-WROTE->Paper`.
6. `Memory` button + stats (`nodes/edges/memories`) prove Track 2.

## Troubleshooting

- `Backend offline. Run uvicorn...`: backend not running or wrong URL. Ensure step 5 terminal is running, frontend calls relative `/api/*` so must open via `http://localhost:8000`, not file://.
- `mode: memory` instead of `falkordb`: Docker/FalkorDB not running. Run `docker compose up -d`, restart uvicorn.
- `Live search came up empty`: S2/OpenAlex rate-limited. Wait 5s, rephrase, try again. Cache is 5 min per query in `app/sources.py:84`.
- `mistral fallback`: key missing/invalid or timeout. App still works, `llm: template`.
- Port 8000 busy: `uvicorn app.main:app --reload --port 8001` then open `http://localhost:8001`.
- `arxiv warn`: arXiv can hang ~90s, code caps total to 14s with daemon threads (`app/sources.py:101`), stragglers ignored.
