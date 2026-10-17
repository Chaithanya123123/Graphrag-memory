"""Concurrency + tenant isolation demo.

Fires 20 parallel /api/ask calls across 3 tenants, then proves no tenant can
read another tenant's memory, episodes or graph.

    python -m uvicorn app.main:app --port 8000
    python scripts/concurrency_demo.py
"""
import concurrent.futures as cf
import sys
import time

import httpx

BASE = "http://localhost:8000"
TENANTS = ["tenant_alpha", "tenant_beta", "tenant_gamma"]
QUESTIONS = [
    "find papers on graph retrieval augmented generation",
    "which claims contradict each other?",
    "most influential paper in this topic",
    "what did I ask before?",
]


def ask(client, user, question):
    t0 = time.time()
    try:
        r = client.post("/api/ask", json={"user_id": user, "question": question})
        return user, r.status_code, time.time() - t0, r.text[:200]
    except Exception as e:
        return user, 599, time.time() - t0, str(e)[:200]


def main():
    n_calls = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    print(f"firing {n_calls} parallel /api/ask calls across {len(TENANTS)} tenants\n")

    markers = {}
    jobs = []
    with httpx.Client(base_url=BASE, timeout=60) as c:
        for i in range(n_calls):
            user = TENANTS[i % len(TENANTS)]
            q = f"{QUESTIONS[i % len(QUESTIONS)]} marker-{user}-{i}"
            jobs.append((user, q))
            markers.setdefault(user, []).append(f"marker-{user}-{i}")

        start = time.time()
        with cf.ThreadPoolExecutor(max_workers=n_calls) as ex:
            results = list(ex.map(lambda j: ask(c, *j), jobs))
        elapsed = time.time() - start

    codes = [code for _, code, _, _ in results]
    lat = sorted(d for _, _, d, _ in results)
    ok = sum(1 for c in codes if c == 200)
    print(f"status codes: {ok}/{n_calls} returned 200")
    p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))]
    print(f"wall clock: {elapsed:.2f}s | p50 {lat[len(lat)//2]:.2f}s | p95 {p95:.2f}s | max {lat[-1]:.2f}s")
    bad = [(u, c, t) for u, c, _, t in results if c != 200]
    for u, c, t in bad[:5]:
        print(f"  FAIL {u} -> {c}: {t}")

    print("\n--- tenant isolation ---")
    leaks = 0
    with httpx.Client(base_url=BASE, timeout=60) as c:
        for user in TENANTS:
            r = c.post("/api/ask", json={"user_id": user, "question": "what did I ask before?"})
            text = r.json().get("text", "")
            mem = r.json().get("memory", [])
            foreign = [o for o in TENANTS if o != user
                       for mk in markers[o] if mk in text or any(mk in m for m in mem)]
            own = sum(1 for mk in markers[user] if mk in text or any(mk in m for m in mem))
            status = "LEAK" if foreign else "clean"
            if foreign:
                leaks += 1
            print(f"  {user:16s} own={own:2d} foreign={len(foreign)} -> {status}")

            g = c.get("/api/graph", params={"user_id": user}).json()
            blob = str(g)
            foreign_nodes = [o for o in TENANTS if o != user and o in blob]
            if foreign_nodes:
                leaks += 1
                print(f"    graph leak: found {foreign_nodes}")

    print(f"\n{'PASS' if leaks == 0 and ok == n_calls else 'FAIL'}: "
          f"{leaks} cross-tenant leak(s), {n_calls - ok} failed call(s)")
    return 0 if (leaks == 0 and ok == n_calls) else 1


if __name__ == "__main__":
    raise SystemExit(main())