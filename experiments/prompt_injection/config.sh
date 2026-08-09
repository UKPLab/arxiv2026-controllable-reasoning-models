#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Configuration for the Green et al. (2025) reasoning-trace extraction attack.
#
# This file is SOURCED by run_attack.sh. It defines, per model, the base model,
# the IF-RT and IF-FA LoRA adapters, and the quantization backend; and, per
# benchmark, the HF dataset and its field names.
# ---------------------------------------------------------------------------

# Model keys used across params.txt and run_attack.sh
MODEL_KEYS=(qwen3_1.7b qwen3_4b qwen3_8b qwen3_14b phi4_3.8b phi4_14b)

# --- Family (drives decoding params + system-prompt compatibility) ----------
declare -A FAMILY=(
  [qwen3_1.7b]=qwen3 [qwen3_4b]=qwen3 [qwen3_8b]=qwen3 [qwen3_14b]=qwen3
  [phi4_3.8b]=phi4   [phi4_14b]=phi4
)

# --- Base models  -----
declare -A BASE_MODEL=(
  [qwen3_1.7b]="unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
  [qwen3_4b]="unsloth/Qwen3-4B-unsloth-bnb-4bit"
  [qwen3_8b]="unsloth/Qwen3-8B-unsloth-bnb-4bit"
  [qwen3_14b]="unsloth/Qwen3-14B-unsloth-bnb-4bit"
  [phi4_3.8b]="unsloth/Phi-4-mini-reasoning-unsloth-bnb-4bit"
  [phi4_14b]="microsoft/Phi-4-reasoning"               # loaded in 4-bit via bitsandbytes
)

# --- Quantization backend passed to --quantization --------------------------
# The inference code uses plain unsloth 4-bit unless the model name lacks
# "unsloth", in which case --quantization bitsandbytes triggers bnb 4-bit.
declare -A QUANT=(
  [qwen3_1.7b]="none" [qwen3_4b]="none" [qwen3_8b]="none" [qwen3_14b]="none"
  [phi4_3.8b]="none"  [phi4_14b]="bitsandbytes"
)

# --- LoRA adapters selected on the dev set (MathIF-GSM8K) -------------------
# IF-RT chkpt.  = highest RT-IF checkpoint (used for the RT stage)
# IF-FA chkpt.  = highest FA-IF checkpoint (used for the FA stage)
# These are the two adapters Staged Decoding switches between
declare -A IFRT_ADAPTER=(
  [qwen3_1.7b]="haritzpuerto/unsloth-Qwen3-1.7B-IF-RT"
  [qwen3_4b]="haritzpuerto/unsloth-Qwen3-4B-IF-RT"
  [qwen3_8b]="haritzpuerto/unsloth-Qwen3-8B-IF-RT"
  [qwen3_14b]="haritzpuerto/unsloth-Qwen3-14B-IF-RT"
  [phi4_3.8b]="haritzpuerto/unsloth-Phi-4-3.8B-IF-RT"
  [phi4_14b]="haritzpuerto/microsoft-Phi-4-14B-IF-RT"
)
declare -A IFFA_ADAPTER=(
  [qwen3_1.7b]="haritzpuerto/unsloth-Qwen3-1.7B-IF-FA"
  [qwen3_4b]="haritzpuerto/unsloth-Qwen3-4B-IF-FA"
  [qwen3_8b]="haritzpuerto/unsloth-Qwen3-8B-IF-FA"
  [qwen3_14b]="haritzpuerto/unsloth-Qwen3-14B-IF-FA"
  [phi4_3.8b]="haritzpuerto/unsloth-Phi-4-3.8B-IF-FA"
  [phi4_14b]="haritzpuerto/microsoft-Phi-4-14B-IF-FA"
)

# --- Decoding params per family (from the paper's Appendix C) ---------------
# Qwen 3: temp 0.6, top-p 0.95, top-k 20, min-p 0
# Phi 4 : temp 0.8, top-p 0.95, top-k 50
qwen3_DECODE="--temperature 0.6 --top-p 0.95 --top-k 20 --min-p 0"
phi4_DECODE="--temperature 0.8 --top-p 0.95 --top-k 50"

# Neither Qwen3 nor Phi4 is incompatible with system prompts (unlike R1).
declare -A INCOMPATIBLE_SYS=(
  [qwen3_1.7b]=0 [qwen3_4b]=0 [qwen3_8b]=0 [qwen3_14b]=0
  [phi4_3.8b]=0  [phi4_14b]=0
)


declare -A BENCH_DATASET=(
  [passwordeval]="haritzpuerto/password_eval-contextual-integrity"
  [peep]="haritzpuerto/PEEP-contextual-integrity"
)
declare -A BENCH_PROMPT_FIELD=(
  [passwordeval]="user_prompt"
  [peep]="user_prompt"
)
declare -A BENCH_SYS_FIELD=(
  [passwordeval]="system_prompt"
  [peep]="system_prompt"
)
# Use in-context learning demonstrations? (1=yes -> reads data/demonstrations/<repo>/demonstration.txt)
declare -A BENCH_USE_ICL=(
  [passwordeval]=1
  [peep]=1
)
declare -A BENCH_EVAL_MODULE=(
  [passwordeval]="password_eval"
  [peep]="peep"
)

# --- Shared inference settings ---------------------------------------------
MAX_TOKENS=32768          # attack output can be ~2x an RT; keep generous
BATCH_SIZE=12             # matches paper
THINK_START="<think>"
THINK_END="</think>"
ATTACK_FILE="data/attacks/green_et_al_injection.txt"

# Where all attack outputs go (override with $ATTACK_OUT_ROOT)
ATTACK_OUT_ROOT="${ATTACK_OUT_ROOT:-runs/prompt_injection_attack}"

# Two seeds
SEEDS=(41875 15613)
