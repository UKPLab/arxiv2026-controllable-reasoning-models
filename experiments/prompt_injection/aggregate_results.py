#!/usr/bin/env python3
"""
Aggregate the prompt-injection attack results into a comparison table.

Reads all ``<tag>_attack_eval.json`` files produced by run_attack.sh and reports,
per (benchmark, model, variant), the final-answer privacy under attack averaged
over seeds. The final-answer channel is what the Green et al. (2025) attack
exfiltrates the (hidden) reasoning trace into, so a higher FA-privacy under
attack means the attack recovered less private content.

Usage:
    python experiments/prompt_injection/aggregate_results.py [ROOT]
    (ROOT defaults to runs/prompt_injection_attack)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/prompt_injection_attack")

# Which metric represents "private content in the FINAL ANSWER" per benchmark.
# (higher = better = less leaked into the visible answer under attack)
FA_METRIC = {
    "passwordeval": "fa_confidential_privacy",  # confidential info kept out of FA
    "peep": "fa_privacy",                       # confidential info kept out of FA
}
# Secondary: password kept out of FA (PasswordEval only) and RT privacy.
EXTRA_METRICS = {
    "passwordeval": ["privacy_final", "privacy_cot"],
    "peep": ["thinking_privacy"],
}


def main():
    if not ROOT.exists():
        sys.exit(f"No results dir at {ROOT}")

    # results[(bench, model, variant)][metric] = [values over seeds]
    results = defaultdict(lambda: defaultdict(list))
    for f in ROOT.rglob("*_attack_eval.json"):
        # path: ROOT/<bench>/<model>/seed<seed>/<variant>_attack_eval.json
        try:
            variant = f.name.replace("_attack_eval.json", "")
            model = f.parent.parent.name
            bench = f.parent.parent.parent.name
        except IndexError:
            continue
        data = json.loads(f.read_text())
        for k, v in data.items():
            if isinstance(v, (int, float)):
                results[(bench, model, variant)][k].append(v)

    if not results:
        sys.exit(f"No *_attack_eval.json files found under {ROOT}")

    for bench in sorted({k[0] for k in results}):
        fa = FA_METRIC.get(bench, "fa_privacy")
        cols = [fa] + EXTRA_METRICS.get(bench, [])
        print(f"\n### {bench}  (FA privacy under attack; higher = less leaked)\n")
        header = ["model", "variant"] + cols + ["n_seeds"]
        print("| " + " | ".join(header) + " |")
        print("|" + "|".join(["---"] * len(header)) + "|")
        for (b, model, variant) in sorted(results):
            if b != bench:
                continue
            row = [model, variant]
            m = results[(b, model, variant)]
            for c in cols:
                vals = m.get(c, [])
                if vals:
                    s = f"{mean(vals):.2f}" + (f"±{pstdev(vals):.2f}" if len(vals) > 1 else "")
                else:
                    s = "-"
                row.append(s)
            n = len(next(iter(m.values()))) if m else 0
            row.append(str(n))
            print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
