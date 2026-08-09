#!/usr/bin/env python3
"""
Does swapping the LoRA adapter mid-generation cost anything vs. using one adapter?

Both conditions do IDENTICAL work — prefill the prompt once, then greedily decode the
same fixed number of tokens (rt_tokens + fa_tokens) with NO re-prefill:

  SINGLE : one adapter (IF-FA) for the whole generation.
  SWAP   : IF-RT for the first rt_tokens, then set_adapter(IF-FA) and CONTINUE decoding
           the remaining fa_tokens over the same KV cache (swap-and-continue = variant b).

The only difference is the mid-stream `set_adapter` call, so the paired time difference
isolates the swap cost. Uses the existing adapters (no aLoRA training; latency is
weight-independent). 🤗 transformers, manual greedy loop for precise, comparable control.

Noise control: fixed token counts (deterministic work), thorough warmup, the two
conditions are timed back-to-back per prompt with ALTERNATING order across rounds, and we
report per-condition mean±std plus a paired bootstrap 95% CI on (SWAP − SINGLE).

  python experiments/latency_swap/run.py --n 50 --rounds 8
"""
import argparse
import random
import statistics as st
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache
from peft import PeftModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="unsloth/Qwen3-4B-unsloth-bnb-4bit")
    p.add_argument("--lora-rt", default="haritzpuerto/unsloth-Qwen3-4B-IF-RT")
    p.add_argument("--lora-fa", default="haritzpuerto/unsloth-Qwen3-4B-IF-FA")
    p.add_argument("--dataset", default="haritzpuerto/password_eval-contextual-integrity")
    p.add_argument("--prompt-field", default="user_prompt")
    p.add_argument("--system-field", default="system_prompt")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--rt-tokens", type=int, default=1024)
    p.add_argument("--fa-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def prefill(model, ids):
    cache = DynamicCache()
    out = model(input_ids=ids, past_key_values=cache, use_cache=True)
    return out.past_key_values, out.logits[:, -1, :].argmax(-1, keepdim=True), ids.shape[1]


@torch.no_grad()
def decode(model, cache, tok, n, pos):
    dev = tok.device
    for _ in range(n):
        out = model(input_ids=tok, past_key_values=cache,
                    cache_position=torch.tensor([pos], device=dev), use_cache=True)
        tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
        cache = out.past_key_values
        pos += 1
    return cache, pos, tok


def run_single(model, ids, nrt, nfa):
    model.set_adapter("iffa")
    cache, tok, pos = prefill(model, ids)
    decode(model, cache, tok, nrt + nfa, pos)


def run_swap(model, ids, nrt, nfa):
    model.set_adapter("ifrt")
    cache, tok, pos = prefill(model, ids)
    cache, pos, tok = decode(model, cache, tok, nrt, pos)
    model.set_adapter("iffa")                       # <-- the mid-generation swap
    decode(model, cache, tok, nfa, pos)


def timed(fn):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize(); return time.perf_counter() - t0


def main():
    a = parse_args()
    rng = random.Random(a.seed)
    tk = AutoTokenizer.from_pretrained(a.base)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(a.base, quantization_config=bnb,
                                                device_map="cuda", torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, a.lora_rt, adapter_name="ifrt")
    model.load_adapter(a.lora_fa, adapter_name="iffa")
    model.eval()

    ds = load_dataset(a.dataset, split="test")
    prompts = []
    for x in ds.select(range(min(a.n, len(ds)))):
        msgs = [{"role": "system", "content": x[a.system_field]},
                {"role": "user", "content": x[a.prompt_field]}]
        enc = tk.apply_chat_template(msgs, add_generation_prompt=True,
                                     return_tensors="pt", return_dict=True)
        prompts.append(enc["input_ids"].to("cuda"))
    def fmt(sec):
        m, s = divmod(int(sec), 60); h, m = divmod(m, 60)
        return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"

    print(f"Loaded {len(prompts)} prompts. Warming up…", flush=True)
    for _ in range(3):
        run_single(model, prompts[0], 8, 8)
        run_swap(model, prompts[0], 8, 8)

    total_pairs = a.rounds * len(prompts)                 # each pair = 1 single + 1 swap timing
    done = 0
    t_start = time.perf_counter()
    print(f"Timing {total_pairs} paired measurements "
          f"({len(prompts)} prompts x {a.rounds} rounds)…", flush=True)
    single_t, swap_t, deltas = [], [], []
    for r in range(a.rounds):
        for i, ids in enumerate(prompts):
            if (r + i) % 2 == 0:                          # alternate order to cancel drift
                ts = timed(lambda: run_single(model, ids, a.rt_tokens, a.fa_tokens))
                tw = timed(lambda: run_swap(model, ids, a.rt_tokens, a.fa_tokens))
            else:
                tw = timed(lambda: run_swap(model, ids, a.rt_tokens, a.fa_tokens))
                ts = timed(lambda: run_single(model, ids, a.rt_tokens, a.fa_tokens))
            single_t.append(ts); swap_t.append(tw); deltas.append(tw - ts)
            done += 1
            elapsed = time.perf_counter() - t_start       # heartbeat + ETA every pair
            eta = elapsed / done * (total_pairs - done)
            bar = "█" * (30 * done // total_pairs)
            print(f"    [{done:>4}/{total_pairs}] {done/total_pairs*100:4.0f}% "
                  f"|{bar:<30}| elapsed {fmt(elapsed)}  eta {fmt(eta)}", flush=True)
        print(f"  round {r}: single={st.mean(single_t[-len(prompts):])*1000:.0f}ms "
              f"swap={st.mean(swap_t[-len(prompts):])*1000:.0f}ms", flush=True)

    B, n = 10000, len(deltas)
    boot = sorted(sum(rng.choice(deltas) for _ in range(n)) / n for _ in range(B))
    lo, hi = boot[int(.025 * B)] * 1000, boot[int(.975 * B)] * 1000
    md = st.mean(deltas) * 1000
    ms, mw = st.mean(single_t) * 1000, st.mean(swap_t) * 1000
    print(f"\n=== {n} paired measurements ({a.n} prompts x {a.rounds} rounds), "
          f"decode {a.rt_tokens}+{a.fa_tokens} tokens ===")
    print(f"one adapter (SINGLE)    : {ms:.1f} ± {st.pstdev(single_t)*1000:.1f} ms")
    print(f"swap-and-continue (SWAP): {mw:.1f} ± {st.pstdev(swap_t)*1000:.1f} ms")
    print(f"paired Δ (SWAP − SINGLE): {md:+.2f} ms  [95% CI {lo:+.2f}, {hi:+.2f}]  ({md/ms*100:+.2f}%)")
    verdict = "negligible (CI brackets ~0)" if (lo < 0 < hi or abs(md) < 0.05 * ms) else "measurable"
    print(f"verdict: swap overhead is {verdict}")


if __name__ == "__main__":
    main()
