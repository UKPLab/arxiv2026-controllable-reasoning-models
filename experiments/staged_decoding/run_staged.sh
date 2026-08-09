#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Clean STAGED DECODING run (no attack) for ONE (model, benchmark) pair.
#
#   Stage 1: reasoning trace with the IF-RT LoRA adapter -> responses_*.jsonl
#   Stage 2: final answer  with the IF-FA LoRA adapter, continuing from </think>
#   Eval   : contextual-privacy leakage in the reasoning trace and final answer.
#
# This mirrors experiments/prompt_injection/run_attack.sh but WITHOUT the
# prompt-injection (--prompt-injection-file), and it runs staged decoding only
# (no untrained baseline). It reuses that experiment's config.sh for the model,
# adapter, decoding and benchmark definitions so the two stay in sync.
#
# Usage (inside an HTCondor job or interactively):
#   bash experiments/staged_decoding/run_staged.sh <model_key> <benchmark> [seed]
#   e.g. bash experiments/staged_decoding/run_staged.sh qwen3_1.7b passwordeval 41875
# ---------------------------------------------------------------------------
set -euo pipefail

MODEL_KEY="${1:?model_key required, e.g. qwen3_1.7b}"
BENCH="${2:?benchmark required: passwordeval|peep}"
SEED="${3:-41875}"

# --- Repo root + environment ------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# HTCondor batch jobs start with a minimal environment: an explicit `environment`
# in the submit file (without getenv=True) means the job gets NO PATH, so even
# coreutils (mkdir, id) are missing. Establish a base PATH first, then default
# USER/HOME (`:=` assigns without tripping `set -u`).
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin${PATH:+:$PATH}"
: "${USER:=$(id -un)}"
: "${HOME:=/home/$USER}"
export USER HOME

# HuggingFace cache on /fast (no file locking -> SoftFileLock; see cluster skill)
export HF_HOME="${HF_HOME:-/fast/$USER/huggingface}"
export SOFTFILELOCK=1
mkdir -p "$HF_HOME"



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

# Reuse the prompt-injection experiment's config (model/adapter/benchmark defs).
source experiments/prompt_injection/config.sh

# Override the shared BATCH_SIZE for this run: Qwen3-1.7B in 4-bit leaves most of
# a 40GB GPU free, so we submit a large decode batch to keep the GPU busy. vLLM
# still bounds actual concurrency by the KV cache (gpu-memory-utilization=0.9, so
# ~10% headroom), so this cannot OOM from batch size alone. Override with
# $STAGED_BATCH_SIZE if a run gets scheduled on a smaller card.
BATCH_SIZE="${STAGED_BATCH_SIZE:-64}"

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

# Where clean staged-decoding outputs go (override with $STAGED_OUT_ROOT)
STAGED_OUT_ROOT="${STAGED_OUT_ROOT:-runs/staged_decoding}"
OUT="$STAGED_OUT_ROOT/$BENCH/$MODEL_KEY/seed$SEED"
mkdir -p "$OUT"

echo "=========================================================="
echo " Staged decoding: model=$MODEL_KEY  bench=$BENCH  seed=$SEED"
echo " base=$BASE  quant=$Q  family=$FAM"
echo " if-rt=$IFRT"
echo " if-fa=$IFFA"
echo " dataset=$DATASET  out=$OUT"
echo "=========================================================="

# Shared inference args for the RT stage (chat over the HF benchmark, no attack).
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
)

# ===========================================================================
# 1) STAGED DECODING - Stage 1: reasoning trace with the IF-RT adapter.
#    Writes responses_{thinking,final_ans,conversations}.jsonl into $STAGE1.
# ===========================================================================
echo "[1/3] Stage 1: reasoning trace (IF-RT adapter)..."
STAGE1="$OUT/stage1"
mkdir -p "$STAGE1"
python -m inference \
  --model "$BASE" \
  --lora-path "$IFRT" \
  "${common_infer_args[@]}" \
  --output-file "$STAGE1/responses.jsonl"

# ===========================================================================
# 2) STAGED DECODING - Stage 2: final answer with the IF-FA adapter, continuing
#    from </think>. Reads the stage-1 folder (responses_conversations.jsonl +
#    responses_thinking.jsonl); the derived _thinking/_final_ans outputs are
#    written next to $OUT/staged.jsonl.
# ===========================================================================
echo "[2/3] Stage 2: final answer (IF-FA adapter)..."
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
  --output-file "$OUT/staged.jsonl"

# ===========================================================================
# 3) Evaluate contextual-privacy leakage (reasoning trace + final answer).
# ===========================================================================
echo "[3/3] Evaluating leakage ($EVAL_MODULE)..."
# Call the eval function directly (avoids the pre-existing import bug in
# evaluation/peep/cli.py) and write a compact JSON next to the outputs.
python - "$EVAL_MODULE" "$OUT/staged_thinking.jsonl" "$OUT/staged_final_ans.jsonl" "$OUT/staged_eval.json" <<'PY'
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

echo "DONE: $MODEL_KEY / $BENCH / seed $SEED"
