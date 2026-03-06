#!/usr/bin/env python3
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import requests
from typing import List, Dict, Any


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def ensure_dirs():
    Path("results").mkdir(parents=True, exist_ok=True)


def ensure_csv_header(csv_path: Path):
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp",
                "run_id",
                "k",
                "hitrate",
                "mrr",
                "avg_embed_ms",
                "avg_search_ms",
                "avg_total_ms",
                "hitrate_pos",
                "mrr_pos",
                "neg_pass_rate",
            ])

def match_expected(hit: Dict[str, Any], expected_sources: List[str]) -> bool:
    src = str(hit.get("source", "")).strip()
    expected = {s.strip() for s in expected_sources}
    return src in expected

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000", help="API base URL")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--questions", default="eval/questions.jsonl")
    ap.add_argument("--source", default=None, help="optional: force source filter for all queries")
    ap.add_argument("--neg-threshold", type=float, default=0.45, help="NegPass: for negative-control questions, pass if top1 score < threshold")
    args = ap.parse_args()

    ensure_dirs()
    csv_path = Path("results/eval_runs.csv")
    latest_json_path = Path("results/latest_eval.json")
    ensure_csv_header(csv_path)

    per_q = []
    embed_ms_list, search_ms_list, total_ms_list = [], [], []
    hit_flags, rr_list = [], []
    pos_hit_flags, pos_rr_list = [], []
    neg_flags = []  # 1 if negative query returns ANY results, else 0

    for q in read_jsonl(Path(args.questions)):
        qid = q["id"]
        query = q["query"]
        expected = q.get("expected_sources", [])

        payload = {"query": query, "top_k": args.k}
        if args.source is not None:
            payload["source"] = args.source

        r = requests.post(f"{args.api}/query", json=payload, timeout=60)
        r.raise_for_status()
        out = r.json()

        # accept both keys, in case you keep "timing_ms"
        timings = out.get("timings_ms") or out.get("timing_ms") or {}
        embed_ms_list.append(float(timings.get("embed_ms", 0.0)))
        search_ms_list.append(float(timings.get("search_ms", 0.0)))
        total_ms_list.append(float(timings.get("total_ms", 0.0)))

        results = out.get("results", []) or []

        hit = any(match_expected(h, expected) for h in results) if expected else False
        hit_flags.append(1 if hit else 0)

        rr = 0.0
        if expected:
            for i, h in enumerate(results, start=1):
                if match_expected(h, expected):
                    rr = 1.0 / i
                    break
        rr_list.append(rr)

        if expected:
            pos_hit_flags.append(1 if hit else 0)
            pos_rr_list.append(rr)
        else:
            top1_score = float(results[0].get("score", 1.0)) if results else 0.0
            neg_pass = 1 if top1_score < args.neg_threshold else 0
            neg_flags.append(neg_pass)

        per_q.append({
            "id": qid,
            "query": query,
            "expected_sources": expected,
            "hit": bool(hit),
            "rr": rr,
            "timings_ms": timings,
            "topk": [
                {
                    "id": h.get("id"),
                    "score": h.get("score"),
                    "source": h.get("source"),
                    "page": h.get("page"),
                    "chunk_idx": h.get("chunk_idx"),
                    "text_preview": h.get("text_preview"),
                }
                for h in results
            ],
        })

    def avg(xs):
        return sum(xs) / max(1, len(xs))

    hitrate = avg(hit_flags)
    mrr = avg(rr_list)

    pos_hitrate = avg(pos_hit_flags) if pos_hit_flags else 0.0
    pos_mrr = avg(pos_rr_list) if pos_rr_list else 0.0

    neg_pass_rate = avg(neg_flags) if neg_flags else 0.0

    report = {
        "run_id": args.run_id,
        "k": args.k,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "neg_threshold": args.neg_threshold,
        "metrics": {
            "HitRate@k": hitrate,
            "MRR@k": mrr,
            "HitRate_pos@k": pos_hitrate,
            "MRR_pos@k": pos_mrr,
            "NegPassRate@k": neg_pass_rate,
        },
        "avg_timings_ms": {
            "embed_ms": avg(embed_ms_list),
            "search_ms": avg(search_ms_list),
            "total_ms": avg(total_ms_list),
        },
        "per_question": per_q,
    }

    latest_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    ts = datetime.now(timezone.utc).isoformat()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            ts, args.run_id, args.k,
            f"{hitrate:.6f}", f"{mrr:.6f}",
            f"{avg(embed_ms_list):.3f}", f"{avg(search_ms_list):.3f}", f"{avg(total_ms_list):.3f}", f"{pos_hitrate:.6f}", f"{pos_mrr:.6f}", f"{neg_pass_rate:.6f}",
        ])

    print(f"Saved: {latest_json_path}")
    print(f"Appended: {csv_path}")
    print(f"HitRate@{args.k}={hitrate:.3f}  MRR@{args.k}={mrr:.3f}")
    print(f"Avg total_ms={avg(total_ms_list):.2f}")


if __name__ == "__main__":
    main()