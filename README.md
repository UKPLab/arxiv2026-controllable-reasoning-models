## Controllable Reasoning Models Are Private Thinkers

[![Hugging Face Models](https://img.shields.io/badge/Hugging%20Face-Models-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/collections/haritzpuerto/controllable-reasoning-models-checkpoints)
[![Hugging Face Datasets](https://img.shields.io/badge/Hugging%20Face-%20Data-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/collections/haritzpuerto/controllable-reasoning-models-datasets)

This repository contains the code and experimental pipelines for the paper **“Controllable Reasoning Models Are Private Thinkers”**.  
The project studies how to train and run large reasoning models so that their **reasoning traces follow instructions** and thereby **reduce contextual privacy leaks** according to a privacy specification instruction while maintaining task utility.

![Figure 1: Reasoning traces of user agents often include private data unnecessary for the task. Through prompt injections, a malicious third-party agent can force the user agent to leak this trace. Instructing the reasoning traces to follow privacy directives is critical to preventing privacy leaks.](static/images/teaser.png)

This teaser figure illustrates how current reasoning models tend to reproduce confidential information and passwords in their reasoning traces, even under explicit privacy-preserving instructions.

In this project we fine-tune LRMs via LoRA to make them follow instructions in their reasoning traces, so that they can follow privacy directives and hence, reduce the possibility of data leaks. The following figure shows the desired behavior we want to achieve.

![Figure 2: Example of contextual information protected by a password, showing desired behavior in green and privacy leaks in red.](static/images/figure1.png)

### Project Overview

- **Goal**: Improve the **instruction-following behavior** of large reasoning models (LRMs) both in their **reasoning traces** and **final answers**, and study how this improves **contextual privacy**.
- **Core idea**:
  - Train models with explicit instructions about how to reason.
  - Use a **staged decoding strategy** that separates reasoning-trace generation and final-answer generation (with different LoRA weights).
- **What this repo provides**:
  - `training/`: fine-tuning code (via Unsloth + TRL) to obtain instruction-following reasoning models.
  - `inference/`: inference pipelines (vLLM) to generate reasoning traces and final answers on multiple benchmarks.
  - `evaluation/`: evaluation scripts for instruction-following and contextual-privacy benchmarks (MathIF, IFEval, PEEP, and PasswordEval).

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

# 6) (Optional) Load additional environment variables (e.g., HF_TOKEN, paths)
# Some experiments use **Hugging Face Hub** models or datasets; We use the script `load_env.sh` for that
source load_env.sh 2>/dev/null || echo "No load_env.sh found or not needed."
```

**Notes**:

- You will need a **GPU with sufficient memory** for training and most inference experiments (e.g., A100 or similar).

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
        --input_data data/ifeval/test.jsonl
        --input_response_data runs/ifeval/sft_thinking.jsonl \
        --output_dir runs/ifeval/sft_thinking \
        --language en

python -m ifeval.cli \
        --input_data
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
  --privacy-evaluation \
  --thinking-path runs/peep/sft_thinking.jsonl \
  --final-response-path runs/peep/sft_final.jsonl \
  --print-stats
```

For the **utility evaluation**, please open the notebook in `src/evaluation/peep/utility_evaluation.ipynb`

**Expected results**:

- Privacy metrics show reduced leakage for sft models compared to baselines.

---

## Third-Party Resources

This project builds on and evaluates against several existing datasets and benchmarks, including but not limited to:

- **MathIF**: instruction-following evaluation for mathematical reasoning.
- **IFEval** for general instruction following.
- **PasswordEval**: contextual-privacy benchmark focusing on password leaks.
- **PEEP**: privacy-focused evaluation of general tasks.

Please refer to our publication for more details on the models and datasets and how we use them. Also check the original publications of those resources for further details.
All dataset usage in this project follows the terms described in the respective papers and licenses (see also our paper appendix).

---

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
TODO
```

Figures 1 and 2 have been designed using resources from Flaticon.com

---

## Experimental Software Disclaimer

This repository contains **experimental software** and is published **for the sole purpose of giving additional background details on the respective publication**.  
It is **not** intended for production use. Results may change as dependencies and models evolve. Please use it at your own risk and always double-check critical outcomes.
