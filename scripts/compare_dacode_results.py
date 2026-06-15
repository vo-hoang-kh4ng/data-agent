"""
Compare two DA-Code eval result JSONs (e.g. unified-global vs per-subtask).

Answers the discovery-difficulty question:
  global lake (172 files, real discovery)  vs  per-subtask lake (trivial discovery).
For each task that flips 0→nonzero under per-subtask, discovery was the blocker in the
global setting. Tasks that stay 0 under per-subtask failed on code-generation, not discovery.

Usage:
    python scripts/compare_dacode_results.py \
        --a data/dacode_official_eval_results.json --a_name "global(172)" \
        --b data/dacode_persubtask_eval_results.json --b_name "per-subtask"
"""

import argparse
import json
from collections import defaultdict


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def by_field(results, field):
    agg = defaultdict(list)
    for r in results:
        agg[r.get(field, "unknown")].append(r["score"])
    return {k: (sum(v) / len(v), len(v)) for k, v in agg.items()}  # mean, count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--a_name", default="A")
    ap.add_argument("--b_name", default="B")
    args = ap.parse_args()

    a, b = load(args.a), load(args.b)
    ra = {r["task_id"]: r for r in a["results"]}
    rb = {r["task_id"]: r for r in b["results"]}
    ids = sorted(set(ra) & set(rb))

    print("=" * 70)
    print(f"  {args.a_name}  vs  {args.b_name}   ({len(ids)} shared tasks)")
    print("=" * 70)
    print(f"  {'':14} {'avg score':>10} {'perfect':>9}")
    print(f"  {args.a_name:14} {a['average_score']:>10.4f} {a['perfect']:>4}/{a['num_results']}")
    print(f"  {args.b_name:14} {b['average_score']:>10.4f} {b['perfect']:>4}/{b['num_results']}")
    delta = b["average_score"] - a["average_score"]
    print(f"  delta (B-A)    {delta:>+10.4f}  ({delta/max(1e-9,a['average_score'])*100:>+5.1f}%)")
    print()

    for field, label in [("category", "category"), ("hardness", "hardness"), ("eval_type", "eval type")]:
        ca = by_field(a["results"], field)
        cb = by_field(b["results"], field)
        keys = sorted(set(ca) | set(cb))
        print(f"  By {label}:")
        print(f"    {'':24} {args.a_name:>12} {args.b_name:>12} {'delta':>9}")
        for k in keys:
            ma, _ = ca.get(k, (0.0, 0))
            mb, _ = cb.get(k, (0.0, 0))
            print(f"    {k:24} {ma:>12.4f} {mb:>12.4f} {mb-ma:>+9.4f}")
        print()

    # Per-task flips (shared tasks only, scoped to A's avg so delta is meaningful)
    a_zero_b_nonzero = [i for i in ids if ra[i]["score"] == 0 and rb[i]["score"] > 0]
    a_nonzero_b_zero = [i for i in ids if ra[i]["score"] > 0 and rb[i]["score"] == 0]
    both_zero = [i for i in ids if ra[i]["score"] == 0 and rb[i]["score"] == 0]
    b_better = [(i, rb[i]["score"] - ra[i]["score"]) for i in ids if rb[i]["score"] > ra[i]["score"]]
    a_better = [(i, ra[i]["score"] - rb[i]["score"]) for i in ids if ra[i]["score"] > rb[i]["score"]]

    print(f"  Discovery was the blocker (A=0 → B>0):  {len(a_zero_b_nonzero)} tasks")
    print(f"     {a_zero_b_nonzero}")
    print(f"  Failed on code-gen, not discovery (A=0 AND B=0): {len(both_zero)} tasks")
    print(f"  B improved (B>A): {len(b_better)} tasks   A better (A>B): {len(a_better)} tasks")
    if b_better:
        top = sorted(b_better, key=lambda x: -x[1])[:10]
        print(f"  Top 10 gains (B-A): {[(i, round(d,3)) for i,d in top]}")
    print("=" * 70)


if __name__ == "__main__":
    main()
