#!/usr/bin/env python3
"""
Aggregate the Green et al. (2025) prompt-injection attack results into a table.

Question: under the RT-extraction attack, does Staged Decoding reduce the private
content pulled into the FINAL ANSWER vs the baseline? => improvement in FA privacy.

FA-privacy metric per benchmark (higher = less leaked into the visible answer):
  passwordeval: privacy_final (PASSWORD in FA)  -- the clean leak signal; the
                password must never appear. (confidential-in-FA is utility-
                confounded: revealing it with the correct password is intended.)
  peep        : fa_privacy    (confidential info in FA -- always a violation)

Usage: python experiments/prompt_injection/aggregate_attack.py [runs/prompt_injection_attack]
"""
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/prompt_injection_attack")
MODELS = ["qwen3_1.7b", "qwen3_4b", "qwen3_8b", "qwen3_14b", "phi4_3.8b", "phi4_14b"]
# benchmark -> primary FA-privacy key (higher = fewer leaks into the final answer)
FA_KEY = {"passwordeval": "privacy_final", "peep": "fa_privacy"}


def load(bench, model, variant):
    p = ROOT / bench / model / "seed41875" / f"{variant}_attack_eval.json"
    return json.loads(p.read_text()) if p.exists() else None


def fmt(x):
    return f"{x:.1f}" if isinstance(x, (int, float)) else "—"


def main():
    for bench in ("passwordeval", "peep"):
        key = FA_KEY[bench]
        extra = bench == "passwordeval"
        sig = "password-in-FA" if bench == "passwordeval" else "confidential-in-FA"
        print(f"\n### {bench} — FA privacy under attack ({sig}; higher = fewer leaks)\n")
        cols = ["Model", "Baseline Priv.FA", "Staged Priv.FA", "Δ (staged−base)", "Leak reduced?"]
        if extra:
            cols += ["Base conf-FA*", "Staged conf-FA*", "Δ conf-FA*"]
        print("| " + " | ".join(cols) + " |")
        print("|" + "|".join(["---"] * len(cols)) + "|")
        deltas = []
        for m in MODELS:
            b, s = load(bench, m, "baseline"), load(bench, m, "staged")
            if b is None and s is None:
                continue
            bv = b.get(key) if b else None
            sv = s.get(key) if s else None
            d = (sv - bv) if (bv is not None and sv is not None) else None
            if d is not None:
                deltas.append(d)
            row = [m, fmt(bv), fmt(sv), (f"{d:+.1f}" if d is not None else "—"),
                   ("✓" if (d or 0) > 0 else ("✗" if d is not None else "—"))]
            if extra:
                bp = b.get("fa_confidential_privacy") if b else None
                sp = s.get("fa_confidential_privacy") if s else None
                dp = (sp - bp) if (bp is not None and sp is not None) else None
                row += [fmt(bp), fmt(sp), (f"{dp:+.1f}" if dp is not None else "—")]
            print("| " + " | ".join(row) + " |")
        if deltas:
            avg = sum(deltas) / len(deltas)
            print(f"\nMean Δ Priv.FA (staged−baseline): {avg:+.1f}  "
                  f"({sum(x>0 for x in deltas)}/{len(deltas)} models improved)")
    print("\n* conf-FA = confidential-info-in-FA: utility-confounded on PasswordEval "
          "(revealing it with the correct password is intended), shown only for context.")


if __name__ == "__main__":
    main()
