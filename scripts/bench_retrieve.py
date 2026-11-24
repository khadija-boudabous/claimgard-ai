"""Bench qualité + latence de /retrieve (stdlib uniquement, lancé depuis l'hôte).

Usage : python3 scripts/bench_retrieve.py --label baseline [--out results.json]
"""

import argparse
import json
import statistics
import time
import urllib.request
from typing import Any

# (question, sous-chaîne du contrat attendu, préfixe de l'article attendu)
GOLDEN: list[tuple[str, str, str]] = [
    ("plafond de remboursement des lunettes", "Santé", "Article 4 "),
    ("délai de carence pour les soins dentaires", "Santé", "Article 2 "),
    ("mon pare-brise est fissuré, suis-je couvert ?", "Auto", "Article 7 "),
    ("fuite d'eau qui endommage l'appartement du voisin", "Habitation", "Article 4 "),
    ("vol sans effraction dans ma maison", "Habitation", "Article 9 "),
]
WARMUP = 5
RUNS = 30


def call(url: str, query: str, top_k: int) -> tuple[dict[str, Any], float]:
    payload = json.dumps({"query": query, "top_k": top_k}).encode()
    req = urllib.request.Request(url, payload, {"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.load(resp)
    return body, (time.perf_counter() - start) * 1000


def rank_of(results: list[dict[str, Any]], contract: str, article: str) -> int | None:
    for i, r in enumerate(results, 1):
        if contract in r["contract"] and r["article"].startswith(article):
            return i
    return None


def pct(values: list[float]) -> dict[str, float]:
    q = statistics.quantiles(values, n=20)  # q[9] = p50, q[18] = p95
    return {"p50": round(q[9], 2), "p95": round(q[18], 2)}


def quality(url: str, top_k: int) -> dict[str, Any]:
    rows, rr = [], []
    for question, contract, article in GOLDEN:
        body, _ = call(url, question, top_k)
        rank = rank_of(body["results"], contract, article)
        rr.append(1 / rank if rank else 0.0)
        top = [f"{r['contract'].split()[-1]} | {r['article'][:30]}" for r in body["results"]]
        rows.append({"question": question, "rank": rank, "top": top})
    hit1 = sum(1 for r in rows if r["rank"] == 1) / len(rows)
    return {"hit@1": hit1, f"MRR@{top_k}": round(statistics.mean(rr), 3), "rows": rows}


def latency(url: str, top_k: int) -> dict[str, Any]:
    for i in range(WARMUP):
        call(url, GOLDEN[i % len(GOLDEN)][0], top_k)
    samples: dict[str, list[float]] = {"search_ms": [], "rerank_ms": [], "total_ms": [], "client_ms": []}
    for i in range(RUNS):
        body, client_ms = call(url, GOLDEN[i % len(GOLDEN)][0], top_k)
        for key in ("search_ms", "total_ms"):
            samples[key].append(body[key])
        samples["rerank_ms"].append(body.get("rerank_ms", 0.0))
        samples["client_ms"].append(client_ms)
    return {key: pct(values) for key, values in samples.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--url", default="http://localhost:8000/retrieve")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--out")
    args = parser.parse_args()

    report = {"label": args.label, **quality(args.url, args.top_k), "latency": latency(args.url, args.top_k)}
    for row in report["rows"]:
        print(f"[rank={row['rank']}] {row['question']}\n    " + "\n    ".join(row["top"]))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2, ensure_ascii=False))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
