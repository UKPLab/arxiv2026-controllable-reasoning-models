## From Leaky Thoughts to Private Reasoning: Controlling What LRMs Say to Themselves
[![arXiv](https://img.shields.io/badge/arXiv-2602.24210-b31b1b.svg)](https://arxiv.org/abs/2602.24210)
[![Hugging Face Models](https://img.shields.io/badge/Hugging%20Face-Models-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/collections/haritzpuerto/controllable-reasoning-models-checkpoints)
[![Hugging Face Datasets](https://img.shields.io/badge/Hugging%20Face-%20Data-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/collections/haritzpuerto/controllable-reasoning-models-datasets)

This repository contains the code and experimental pipelines for the paper **“From Leaky Thoughts to Private Reasoning: Controlling What LRMs Say to Themselves”**.

Large reasoning models (LRMs) write reasoning traces (RTs) that often reproduce sensitive context — names, phone numbers, passwords — **even when the system prompt forbids it**. These *leaky thoughts* are not safe just because they are hidden: a prompt injection can pull the RT into the visible answer ([Green et al., 2025](https://aclanthology.org/2025.emnlp-main.1347/)). We treat this as a **controllability** problem: a privacy directive is just an instruction, so improving **instruction following inside the RT** (IF-RT) is a direct path to fewer privacy leaks.

![Figure 1: An LRM without controllable reasoning copies the secret and the password into its reasoning trace despite the privacy directive; with controllable reasoning the trace stays private while the final answer is unchanged.](static/images/figure1.png)

### Project Overview

- **Goal**: Improve the **instruction-following behavior** of large reasoning models (LRMs) both in their **reasoning traces** and **final answers**, and study how this improves **contextual privacy**.
- **Core idea**:
  - Train models with explicit instructions about how to reason.
  - Use a **staged decoding strategy** that separates reasoning-trace generation and final-answer generation (with different LoRA weights).

### What This Repo Provides

**Code**

| Directory | Contents |
| --- | --- |
| [src/training/](src/training/) | Fine-tuning code (Unsloth + TRL) to obtain instruction-following reasoning models. |
| [src/inference/](src/inference/) | Inference pipelines (vLLM) that generate reasoning traces and final answers on multiple benchmarks. |
| [src/evaluation/](src/evaluation/) | Evaluation scripts for the instruction-following and contextual-privacy benchmarks (MathIF, IFEval, PEEP, PasswordEval). |
| [src/data_creation/](src/data_creation/) | Notebooks that build the evaluation data for each benchmark. |

**Experiments** — [experiments/](experiments/) holds the cluster scripts and analyses behind the tables in the paper:

- [staged_decoding/](experiments/staged_decoding/) — the two-stage (RT / answer) decoding strategy.
- [prompt_injection/](experiments/prompt_injection/) — attacks that pull the reasoning trace into the answer.
- [latency_swap/](experiments/latency_swap/) — inference-cost measurements with LoRA swapping.
- [judge_eval/](experiments/judge_eval/) — quality control for the LLM judge.
- [significance/](experiments/significance/) — statistical tests on the reported results.

---

## Installation and Setup

The following script creates a fresh virtual environment and installs all dependencies locally.  
You can copy-paste it directly into your shell (Linux/macOS):

```bash
# 1) Clone the repository
git clone https://github.com/UKPLab/arxiv2026-controllable-reasoning-models
cd arxiv2026-controllable-reasoning-models

# 2) Create and activate a fresh virtual environment (Python >= 3.12)
# uv automatically handles fetching the right Python version if you don't have it
uv venv --python 3.12
source .venv/bin/activate

# 3) Upgrade pip (Optional with uv)
# uv manages its own binaries, but if you need the latest pip inside the env:
uv pip install --upgrade pip

# 4) Install the project in editable mode + all dependencies
uv pip install -e . -r requirements.txt

# 5) (Optional but recommended) Set up local Hugging Face cache if you lack global write permissions (common in HPC)
export HF_HOME="$(pwd)/.cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TMPDIR="$HF_HOME/tmp"
mkdir -p .cache/{datasets,hub,tmp}

# 6) Load additional environment variables (e.g., HF_TOKEN, paths)
# Some experiments use **Hugging Face Hub** models or datasets; We use the script `load_env.sh` for that
source load_env.sh 2>/dev/null || echo "No load_env.sh found or not needed."
```

**Notes**:

- You will need a **GPU with sufficient memory** for training and most inference experiments (all our experiments ran on a single Nvidia A100).

---

## How to Use This Project

This section gives **end-to-end example scripts** to (1) fine-tune a model with our SFT instruction-follwing CoT datset, (2) run inference on evaluation benchmarks, and (3) compute the reported metrics.  
The commands are designed to be copy-paste friendly; adjust paths and model names according to your hardware and data locations.

### 1. Training Models (Unsloth + TRL)

Example (single-GPU fine-tuning with default settings):

```bash
python -m training \
  --dataset "haritzpuerto/instruction-following-reasoning-traces" \
  --split "rt_only" \
  --model_path "Qwen/Qwen3-1.7B" \
  --max_seq_length 3100 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --output_dir "outputs/sft-Qwen3-1.7B"
```

**Expected results**:

- Training logs (e.g., via `wandb` if enabled) will show decreasing loss and convergence of instruction-following metrics on the training data.
- The resulting LoRA/adapter weights in `outputs/sft-Qwen3-1.7B` are the **sft models** used for the experiments in the paper.

### 2. Inference: Generating Reasoning Traces and Final Answers

Use the `inference` module to generate outputs on benchmarks (e.g., IFEval, MathIF, PEEP, and PasswordEval.).

Example: run inference with vLLM on an instruction-following benchmark:

```bash
python -m inference \
  --model "Qwen/Qwen3-1.7B" \
  --lora-path "outputs/sft-Qwen3-1.7B" \
  --dataset hf \
  --data-file haritzpuerto/ifeval-lrm \
  --prompt-field "prompt" \
  --output-file runs/ifeval/sft-Qwen3-1.7B.jsonl \
  --dataset ifr \
  --batch-size 8 \
  --max-tokens 512 \
  --temperature 0.7 \
  --top-p 0.9 \
  --think-token-start "<think>" \
  --think-token-end "</think>"
```

**Expected results**:

- `runs/ifeval/sft-Qwen3-1.7B` will contain both **reasoning traces** and **final answers**.
- These files are the inputs to the evaluation scripts below.

### 3. Evaluation: Instruction Following and Contextual Privacy

The `evaluation/` package contains several task-specific CLIs.  
All of them follow the same pattern: **provide paths to the benchmark data and the generated model outputs**.

#### 3.1 IFEval

```bash
python -m ifeval.cli \
        --input_data data/ifeval/test.jsonl \
        --input_response_data runs/ifeval/sft_thinking.jsonl \
        --output_dir runs/ifeval/sft_thinking \
        --language en

python -m ifeval.cli \
        --input_data data/ifeval/test.jsonl \
        --input_response_data runs/ifeval/sft_final_ans.jsonl \
        --output_dir runs/ifeval/sft_final_ans \
        --language en
```

#### 3.2 MathIF Instruction-Following

```bash
python -m evaluation.math_if \
  --data-path data/math_if/test.jsonl \
  --thinking-path runs/mathif/sft_thinking.jsonl \
  --final-ans-path runs/mathif/sft_final_ans.jsonl \
  --print-stats
```

**Expected results**:

- The script prints summary statistics of instruction-following performance on MathIF.

#### 3.3 PasswordEval Contextual-Privacy Benchmark

```bash
python -m evaluation.password_eval \
  --thinking-path runs/password_eval/sft_thinking.jsonl \
  --final-response-path runs/password_eval/sft_final.jsonl \
  --print-stats
```

**Expected results**:

- The script reports how often sensitive information is leaked in reasoning traces or final answers, corresponding to the contextual-privacy metrics in the paper.

#### 3.4 PEEP Privacy and Utility Evaluation

```bash
# Privacy evaluation
python -m evaluation.peep \
  --thinking-path runs/peep/sft_thinking.jsonl \
  --final-response-path runs/peep/sft_final.jsonl \
  --print-stats
```

For the **utility evaluation**, please open the notebook in `src/evaluation/peep/utility_evaluation.ipynb`

**Expected results**:

- Privacy metrics show reduced leakage for sft models compared to baselines.

---

## Third-Party Resources

This project builds on datasets, models, and code released by others. Each is listed below with
its source, license, and citation. The derived artifacts we release on the Hugging Face Hub
inherit the license of the resource they are derived from; see each Hub repository for details.

### Datasets and benchmarks

| Resource | Used for | Source | License |
| --- | --- | --- | --- |
| **IFEval** | General instruction following. `data/ifeval/test.jsonl` is IFEval with the instructions restated to apply to the reasoning trace. | [google/IFEval](https://huggingface.co/datasets/google/IFEval) | Apache-2.0 |
| **MathIF** | Instruction following under mathematical reasoning. Items derive from GSM8K, MATH-500, Minerva, OlympiadBench, and AIME. | [TingchenFu/MathIF](https://github.com/TingchenFu/MathIF) | MIT |
| **PasswordEval** | Contextual-privacy benchmark on password-gated confidential information. Our [password_eval-contextual-integrity](https://huggingface.co/datasets/haritzpuerto/password_eval-contextual-integrity) adds privacy directives to the original system prompts. | [locuslab/password_eval](https://huggingface.co/datasets/locuslab/password_eval) | CC BY 4.0 |
| **PEEP** | Contextual-privacy and utility evaluation on real user queries. Our [PEEP-contextual-integrity](https://huggingface.co/datasets/haritzpuerto/PEEP-contextual-integrity) adds privacy directives; the utility judge-validation exports in [experiments/judge_eval/](experiments/judge_eval/) contain derived rows. | [guillemram97/PEEP](https://huggingface.co/datasets/guillemram97/PEEP) | ODC-BY |
| **WildChat** | Upstream source of PEEP's user queries. | [allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) | ODC-BY |
| **Multilingual-Thinking** | Default training set in [src/training/cli.py](src/training/cli.py). | [HuggingFaceH4/Multilingual-Thinking](https://huggingface.co/datasets/HuggingFaceH4/Multilingual-Thinking) | Apache-2.0 |

PEEP and WildChat are distributed under ODC-BY, which requires attribution on redistributed
derivatives.

### Base models

| Model | Source | License |
| --- | --- | --- |
| Qwen3 (1.7B / 4B / 8B / 14B), incl. the `unsloth` 4-bit builds | [Qwen](https://huggingface.co/Qwen) · [unsloth](https://huggingface.co/unsloth) | Apache-2.0 |
| Phi-4-reasoning (14B) and Phi-4-mini-reasoning (3.8B), incl. the `unsloth` 4-bit build | [microsoft/Phi-4-reasoning](https://huggingface.co/microsoft/Phi-4-reasoning) | MIT |


### Code

- **Google Research IFEval** — [`src/evaluation/math_if/constraint_checker.py`](src/evaluation/math_if/constraint_checker.py),
  [`constraint_util.py`](src/evaluation/math_if/constraint_util.py), and
  [`constraint_registry.py`](src/evaluation/math_if/constraint_registry.py) are adapted from
  [google-research/instruction_following_eval](https://github.com/google-research/google-research/tree/master/instruction_following_eval)
  (Apache-2.0). The original copyright headers are retained in each file; see [NOTICE](NOTICE).
- **[oKatanaaa/ifeval](https://github.com/oKatanaaa/ifeval)** (Apache-2.0) — packaged IFEval scorer, installed via [requirements.txt](requirements.txt).,
- **Project page** — [index.html](index.html) uses the
  [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template) (CC BY-SA 4.0),
  itself based on [Nerfies](https://nerfies.github.io).


## Institutional Links

- **UKP Lab** (Ubiquitous Knowledge Processing Lab):  
  [https://www.ukp.tu-darmstadt.de/](https://www.ukp.tu-darmstadt.de/)

- **Technische Universität Darmstadt**:  
  [https://www.tu-darmstadt.de/](https://www.tu-darmstadt.de/)

---

## Maintainers and Contact

- **Haritz Puerto**
  - GitHub: [@HaritzPuerto](https://github.com/HaritzPuerto)
  - Website: [https://haritzpuerto.github.io](https://haritzpuerto.github.io)

For questions, bug reports, or feature requests, please send an email to Haritz Puerto. You can find his up-to-date email in his website.

---

## Citation

If you use this code or any of the released models or data, please cite:

```bibtex
@misc{puerto2026leakythoughtsprivatereasoning,
      title={From Leaky Thoughts to Private Reasoning: Controlling What LRMs Say to Themselves}, 
      author={Haritz Puerto and Haonan Li and Xudong Han and Timothy Baldwin and Iryna Gurevych},
      year={2026},
      eprint={2602.24210},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.24210}, 
}
```

---

## Experimental Software Disclaimer

This repository contains **experimental software** and is published **for the sole purpose of giving additional background details on the respective publication**.  
It is **not** intended for production use. Results may change as dependencies and models evolve. Please use it at your own risk and always double-check critical outcomes.
