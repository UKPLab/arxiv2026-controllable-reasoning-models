# Adapter-Swap Latency

Staged Decoding switches LoRA adapters **mid-generation** — IF-RT writes the reasoning
trace, then IF-FA writes the final answer. This micro-benchmark answers the obvious
objection: *does that switch cost anything at inference time?*

Two conditions do **identical** work — one prefill, then a greedy decode of exactly
`rt_tokens + fa_tokens` tokens over the same KV cache, with no re-prefill:

| | |
|---|---|
| `SINGLE` | one adapter (IF-FA) for the whole generation |
| `SWAP`   | IF-RT for the first `rt_tokens`, then `set_adapter("iffa")` and **continue** decoding the remaining `fa_tokens` on the same cache |

The only difference is the mid-stream `set_adapter` call, so the paired time difference
isolates the swap cost. Latency here is weight-independent, so the existing released
adapters are used as-is (no extra training). Implemented as a manual greedy loop in 🤗
transformers rather than `generate()`, so both conditions run precisely the same number
of forward passes.

**Noise control:** fixed token counts (deterministic work per measurement), a warmup
phase, the two conditions timed back-to-back per prompt with **alternating order** across
rounds to cancel thermal/clock drift, and `torch.cuda.synchronize()` around every timing.

## Run

```bash
python experiments/latency_swap/run.py --n 50 --rounds 8 --rt-tokens 1024 --fa-tokens 512
```

Defaults: base `unsloth/Qwen3-4B-unsloth-bnb-4bit` (4-bit), adapters
`haritzpuerto/unsloth-Qwen3-4B-IF-{RT,FA}`, prompts from the PEEP contextual-integrity
test split. Requires a GPU.

## Output

Printed to stdout (nothing is written to disk): per-condition mean ± std over all
`n × rounds` paired measurements, the paired Δ (`SWAP − SINGLE`) in ms and as a percentage
of `SINGLE`, and a **paired bootstrap 95% CI** on that Δ (10k resamples). The verdict line
calls the overhead *negligible* when the CI brackets zero or |Δ| < 5% of `SINGLE`.
