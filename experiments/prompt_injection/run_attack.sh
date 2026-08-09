#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run the Green et al. (2025) reasoning-trace extraction attack for ONE
# (model, benchmark) pair, comparing the untrained BASELINE against STAGED
# DECODING. For each we (1) run inference with the prompt injection appended
# to the user turn, then (2) evaluate leakage in the final answer (the channel
# the attack exfiltrates the hidden reasoning trace into).
#
# Usage (inside an HTCondor job or interactively):
#   bash experiments/prompt_injection/run_attack.sh <model_key> <benchmark> [seed]
#   e.g. bash experiments/prompt_injection/run_attack.sh qwen3_8b passwordeval 41875
# ---------------------------------------------------------------------------
set -euo pipefail

MODEL_KEY="${1:?model_key required, e.g. qwen3_8b}"
BENCH="${2:?benchmark required: passwordeval|peep}"
SEED="${3:-41875}"

# --- Repo root + environment ------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

module load cuda/13.2 

# Triton JIT-compiles CUDA launchers at runtime. It reads the C compiler name
# from the standalone Python's sysconfig, which hardcodes a nonexistent
# 'gcc-4.6' (NOT overridable via $CC). Put a shim named exactly that on PATH,
# pointing at the real gcc, plus set CC/CXX for anything that does honor them.
module load gcc 2>/dev/null || true
SHIM_DIR="$HOME/.local/triton-cc-shim"
mkdir -p "$SHIM_DIR"
if command -v gcc >/dev/null 2>&1; then
  ln -sf "$(command -v gcc)" "$SHIM_DIR/gcc-4.6"
  export CC="$(command -v gcc)"
fi
if command -v g++ >/dev/null 2>&1; then
  ln -sf "$(command -v g++)" "$SHIM_DIR/g++-4.6"
  export CXX="$(command -v g++)"
fi
export PATH="$SHIM_DIR:$PATH"

# Project env (venv + HF_TOKEN etc.)
source .venv/bin/activate 2>/dev/null || true
source load_env.sh 2>/dev/null || true

source experiments/prompt_injection/config.sh

# --- Resolve per-model / per-benchmark config ------------------------------
BASE="${BASE_MODEL[$MODEL_KEY]}"
IFRT="${IFRT_ADAPTER[$MODEL_KEY]}"
IFFA="${IFFA_ADAPTER[$MODEL_KEY]}"
Q="${QUANT[$MODEL_KEY]}"
FAM="${FAMILY[$MODEL_KEY]}"
INCOMPAT="${INCOMPATIBLE_SYS[$MODEL_KEY]}"

DATASET="${BENCH_DATASET[$BENCH]}"
PROMPT_FIELD="${BENCH_PROMPT_FIELD[$BENCH]}"
SYS_FIELD="${BENCH_SYS_FIELD[$BENCH]}"
USE_ICL="${BENCH_USE_ICL[$BENCH]}"
EVAL_MODULE="${BENCH_EVAL_MODULE[$BENCH]}"

# Decoding params for the family (indirect variable expansion)
DECODE_VAR="${FAM}_DECODE"
DECODE="${!DECODE_VAR}"

# Optional flags
ICL_FLAG=""; [ "$USE_ICL" = "1" ] && ICL_FLAG="--use-in-context-learning"
INCOMPAT_FLAG=""; [ "$INCOMPAT" = "1" ] && INCOMPAT_FLAG="--incompatible-with-sys-prompt"

OUT="$ATTACK_OUT_ROOT/$BENCH/$MODEL_KEY/seed$SEED"
mkdir -p "$OUT"

echo "=========================================================="
echo " Attack run: model=$MODEL_KEY  bench=$BENCH  seed=$SEED"
echo " base=$BASE  quant=$Q  family=$FAM"
echo " if-rt=$IFRT"
echo " if-fa=$IFFA"
echo " dataset=$DATASET  out=$OUT"
echo "=========================================================="

common_infer_args=(
  --dataset hf
  --data-file "$DATASET"
  --prompt-field "$PROMPT_FIELD"
  --system-prompt "$SYS_FIELD"
  $ICL_FLAG $INCOMPAT_FLAG
  --batch-size "$BATCH_SIZE"
  --max-tokens "$MAX_TOKENS"
  $DECODE
  --seed "$SEED"
  --quantization "$Q"
  --think-token-start "$THINK_START"
  --think-token-end "$THINK_END"
  --prompt-injection-file "$ATTACK_FILE"
)

# ===========================================================================
# 1) BASELINE under attack (single-pass full generation, no adapter)
# ===========================================================================
echo "[1/3] Baseline inference under attack..."
python -m inference \
  --model "$BASE" \
  "${common_infer_args[@]}" \
  --output-file "$OUT/baseline_attack.jsonl"

# ===========================================================================
# 2) STAGED DECODING under attack
#    Stage 1: RT with IF-RT adapter (+injection) -> writes responses_*.jsonl
#    Stage 2: FA with IF-FA adapter, continuing from </think>
#             (injection is inherited from the stage-1 conversations file)
# ===========================================================================
echo "[2/3] Staged decoding under attack (stage 1: RT / IF-RT adapter)..."
STAGE1="$OUT/staged_stage1"
mkdir -p "$STAGE1"
python -m inference \
  --model "$BASE" \
  --lora-path "$IFRT" \
  "${common_infer_args[@]}" \
  --output-file "$STAGE1/responses.jsonl"   # -> responses_{thinking,final_ans,conversations}.jsonl

echo "[2/3] Staged decoding under attack (stage 2: FA / IF-FA adapter)..."
# Stage 2 reads the folder (responses_conversations.jsonl + responses_thinking.jsonl),
# so it needs NO --prompt-injection flag: the injection is already baked into the
# saved conversations from stage 1.
python -m inference \
  --model "$BASE" \
  --lora-path "$IFFA" \
  --dataset complete_final_ans \
  --data-file "$STAGE1" \
  --batch-size "$BATCH_SIZE" \
  --max-tokens "$MAX_TOKENS" \
  $DECODE \
  --seed "$SEED" \
  --quantization "$Q" \
  --think-token-start "$THINK_START" \
  --think-token-end "$THINK_END" \
  --output-file "$OUT/staged_attack.jsonl"

# ===========================================================================
# 3) Evaluate leakage under attack (final-answer channel is what matters)
# ===========================================================================
echo "[3/3] Evaluating leakage under attack..."
eval_one () {  # <tag> <thinking> <final_ans>
  local tag="$1" think="$2" final="$3"
  echo "--- eval: $tag ($EVAL_MODULE) ---"
  # Call the eval function directly (avoids the pre-existing import bug in
  # evaluation/peep/cli.py) and write a compact JSON next to the outputs.
  python - "$EVAL_MODULE" "$think" "$final" "$OUT/${tag}_attack_eval.json" <<'PY'
import sys, json
mod, think, final, outp = sys.argv[1:5]
if mod == "password_eval":
    from evaluation.password_eval.evaluation import evaluate as ev
    m = ev(thinking_path=think, final_response_path=final, print_stats=True)
elif mod == "peep":
    from evaluation.peep.evaluation import evaluate_leaks as ev
    m = ev(thinking_path=think, final_response_path=final, print_stats=True)
else:
    raise SystemExit(f"unknown eval module: {mod}")
with open(outp, "w") as f:
    json.dump(m, f, indent=2)
print(f"saved -> {outp}")
PY
}

eval_one baseline "$OUT/baseline_attack_thinking.jsonl" "$OUT/baseline_attack_final_ans.jsonl"
eval_one staged   "$OUT/staged_attack_thinking.jsonl"   "$OUT/staged_attack_final_ans.jsonl"

echo "DONE: $MODEL_KEY / $BENCH / seed $SEED"
