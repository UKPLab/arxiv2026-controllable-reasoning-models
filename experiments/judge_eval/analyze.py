#!/usr/bin/env python3
"""
Analyze the human evaluation of the PEEP LLM judge.

Input: one or more JSON exports from annotation.html (batch-1 + batch-2 …). Each record
has judge_overall and, if annotated, human_overall / acceptable / reasons. Merges the
exports (dedup by provenance), then computes human<->judge agreement with metrics suited
to skewed ordinal ratings, plus a per-model breakdown.

  python3 experiments/judge_eval/analyze.py export1.json [export2.json ...]

Pure stdlib (kappa / Spearman implemented by hand).
"""
import json
import sys
from collections import Counter, defaultdict

PATHS = sys.argv[1:] or ["experiments/judge_eval/peep_judge_annotations.json"]
K = 5  # score categories 1..5


def prov_key(d):
    return (d["model"], d["size"], d["variant"], str(d["seed"]), int(d["req_idx"]))


def quadratic_weighted_kappa(human, judge, kmin=1, kmax=K):
    cats = list(range(kmin, kmax + 1))
    idx = {c: i for i, c in enumerate(cats)}
    n = len(cats)
    O = [[0] * n for _ in range(n)]
    for h, j in zip(human, judge):
        O[idx[h]][idx[j]] += 1
    hist_h = [0] * n
    hist_j = [0] * n
    for h, j in zip(human, judge):
        hist_h[idx[h]] += 1
        hist_j[idx[j]] += 1
    N = len(human)
    W = [[((i - j) ** 2) / ((n - 1) ** 2) for j in range(n)] for i in range(n)]
    E = [[hist_h[i] * hist_j[j] / N for j in range(n)] for i in range(n)]
    num = sum(W[i][j] * O[i][j] for i in range(n) for j in range(n))
    den = sum(W[i][j] * E[i][j] for i in range(n) for j in range(n))
    return 1 - num / den if den else float("nan")


def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((y - mb) ** 2 for y in b) ** 0.5
    return cov / (va * vb) if va and vb else float("nan")


def spearman(a, b):
    return pearson(_ranks(a), _ranks(b))


def main():
    # Merge all exports; dedup by provenance (keep first annotated occurrence).
    merged = {}
    for p in PATHS:
        for d in json.loads(open(p).read()):
            if not (d.get("human_overall") and d.get("acceptable") is not None):
                continue
            merged.setdefault(prov_key(d), d)
    ann = list(merged.values())
    if not ann:
        sys.exit(f"No completed annotations in {PATHS} (need human_overall + acceptable).")

    human = [int(d["human_overall"]) for d in ann]
    judge = [int(d["judge_overall"]) for d in ann]
    N = len(ann)

    mae = sum(abs(h - j) for h, j in zip(human, judge)) / N
    within1 = sum(abs(h - j) <= 1 for h, j in zip(human, judge)) / N * 100
    exact = sum(h == j for h, j in zip(human, judge)) / N * 100
    acc = sum(d["acceptable"] is True for d in ann) / N * 100
    qwk = quadratic_weighted_kappa(human, judge)
    rho = spearman(human, judge)

    print(f"# PEEP LLM-judge human evaluation  (N = {N} annotated)\n")
    print("| Metric | Value |")
    print("|---|---|")
    print(f"| Acceptability rate (judge assessment reasonable) | {acc:.0f}% |")
    print(f"| Quadratic-weighted Cohen's κ (human vs judge overall) | {qwk:.2f} |")
    print(f"| Spearman ρ | {rho:.2f} |")
    print(f"| Mean absolute error (|human − judge|) | {mae:.2f} |")
    print(f"| Agreement within ±1 point | {within1:.0f}% |")
    print(f"| Exact agreement | {exact:.0f}% |")

    # per model-size breakdown
    by = defaultdict(list)
    for d in ann:
        by[f"{d['model']}/{d['size']}"].append(d)
    print("\n## By model-size\n")
    print("| Model | n | Acceptable% | MAE | ±1% |")
    print("|---|---|---|---|---|")
    for ms in sorted(by):
        g = by[ms]
        h = [int(x["human_overall"]) for x in g]
        j = [int(x["judge_overall"]) for x in g]
        m = sum(abs(a - b) for a, b in zip(h, j)) / len(g)
        w = sum(abs(a - b) <= 1 for a, b in zip(h, j)) / len(g) * 100
        a_ = sum(x["acceptable"] is True for x in g) / len(g) * 100
        print(f"| {ms} | {len(g)} | {a_:.0f} | {m:.2f} | {w:.0f} |")

    # reasons for "not good"
    reasons = Counter(r for d in ann if d.get("acceptable") is False for r in (d.get("reasons") or []))
    if reasons:
        print("\n## Why judgments were flagged (counts)\n")
        for r, c in reasons.most_common():
            print(f"- {r}: {c}")

    # bias direction
    over = sum(j > h for h, j in zip(human, judge))
    under = sum(j < h for h, j in zip(human, judge))
    print(f"\nDirection: judge scored higher than human {over}×, lower {under}×, equal {N-over-under}×.")

    print("\n---\n## Draft methodology paragraph\n")
    print(
        f"To validate the GPT-5-nano utility judge, one author annotated a random sample of "
        f"{N} English PEEP responses stratified across all six models, assigning an overall "
        f"1–5 score under the judge's rubric and marking whether the judge's assessment was "
        f"reasonable. We validate on the English subset (the annotator's language); the judge "
        f"itself, being multilingual, is applied to all languages. Because the scores are "
        f"ordinal and skewed toward the top of the scale, we report agreement rather than "
        f"accuracy: quadratic-weighted Cohen's κ = {qwk:.2f}, Spearman ρ = {rho:.2f}, mean "
        f"absolute error = {mae:.2f} points, and {within1:.0f}% of judge scores fall within "
        f"±1 of the human score; the human deemed {acc:.0f}% of the judge's assessments "
        f"reasonable. As the judge is only a secondary utility metric (privacy is measured "
        f"deterministically), this level of agreement is sufficient to support our conclusions."
    )


if __name__ == "__main__":
    main()
