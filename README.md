# MedChat: Medical Conversational AI Assistant

[![Base Model](https://img.shields.io/badge/Base_Model-Mistral--7B--Instruct--v0.3-orange.svg)](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20HuggingFace%20%7C%20TRL%20%7C%20PEFT-purple.svg)](https://github.com/huggingface/trl)
[![Hardware](https://img.shields.io/badge/Compute-Kaggle%20T4%20(16GB)%20%2F%20A100-green.svg)]()

> **Clinical & Safety Disclaimer:**  
> This project is an academic research prototype and is **not a certified medical device**. It is designed solely for general health-information education and conversational research. It cannot perform clinical diagnoses, formulate treatment plans, or prescribe medications. In any life-threatening situation or emergency, users must contact local emergency services (e.g., 911 / 112) immediately.

---

## 📌 Executive Summary

This repository implements a multi-turn, privacy-preserving Medical AI Chatbot built on top of **`mistralai/Mistral-7B-Instruct-v0.3`** (Apache-2.0).

While earlier exploratory iterations of this codebase tested Qwen models, this project transitioned to **Mistral-7B-Instruct-v0.3** due to:
1. **Clean Chat Format**: Mistral's native `[INST] ... [/INST]` format eliminates uncalibrated `<think>` token leakage and ChatML parsing inconsistencies.
2. **Clinical Reasoning & Instruction Following**: Mistral-7B demonstrates superior medical reasoning and fact retention compared to 1.7B-scale models.
3. **Training Efficiency**: 4-bit QLoRA (NF4) fits within 16 GB VRAM GPUs (Kaggle T4 / P100), enabling cost-effective fine-tuning and post-SFT alignment.

The training workflow combines **Supervised Fine-Tuning (SFT)** with **Group Relative Policy Optimization (GRPO)**, followed by a clinical evaluation and accuracy testing harness.

---

## 🏗️ System Architecture & Workflow

```
                        ┌─────────────────────────────────────────────────────────┐
                        │ data/processed/integrated_conversations_ner_validated.csv│
                        └───────────────────────────┬─────────────────────────────┘
                                                    │
                                      sft/build_jsonl.py (Data Prep)
                                                    │
                                    ┌───────────────┴───────────────┐
                                    ▼                               ▼
                           data/sft/v1/train.jsonl         data/sft/v1/val.jsonl
                           (90% stratified dialogues)      (10% validation)
                                    │
                                    ▼
                        ┌─────────────────────────────────────────┐
                        │  Phase 1: Supervised Fine-Tuning (SFT)  │
                        │     Base: Mistral-7B-Instruct-v0.3      │
                        │     Method: 4-bit QLoRA (NF4) / fp16    │
                        │     Loss: Supervised on Assistant only  │
                        │     Script: sft/train_sft.py            │
                        └───────────────────┬─────────────────────┘
                                            │ Checkpoint Adapter
                                            ▼
                        ┌─────────────────────────────────────────┐
                        │ Phase 2: Post-SFT RL Alignment (GRPO)   │
                        │     Policy: SFT Adapter Checkpoint      │
                        │     Algorithm: GRPOTrainer (TRL)        │
                        │     Reward: Composite Safety + Style    │
                        │     Script: sft/train_grpo.py           │
                        └───────────────────┬─────────────────────┘
                                            │ Final Aligned Adapter
                                            ▼
                        ┌─────────────────────────────────────────┐
                        │ Phase 3: Clinical Evaluation & Accuracy │
                        │     - Emergency Escalation Rate (EER)   │
                        │     - Clinical MCQ Knowledge Accuracy   │
                        │     - Persona & Lead-Gen Gating         │
                        │     - Script: sft/evaluate.py           │
                        └─────────────────────────────────────────┘
```

---

## 🔬 Core Training Methodology

### 1. Supervised Fine-Tuning (SFT)
- **Tokenization & Supervision**: Formatted using Mistral chat formatting `<s>[INST] {system_prompt}\n\n{user_turn} [/INST] {assistant_turn}</s>`. All user turns and formatting tokens receive `label = -100`. Only the assistant's tokens and closing `</s>` (id=2) contribute to the loss.
- **LoRA Hyperparameters**: Rank $r=16$, $\alpha=32$, dropout $0.05$, applied to all linear projections (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`).
- **Precision**: 4-bit NF4 double quantization with fp16 compute (`use_qlora`) for T4 GPUs; bf16 for Ampere+ (A100).

### 2. Reinforcement Learning via GRPO (Group Relative Policy Optimization)
Traditional RLHF (PPO) requires training a separate critic/value network, which often exceeds the memory limits of a single 16 GB GPU. **GRPO** resolves this by:
1. Sampling $K = 4$ (or $8$) candidate responses $\{y_1, \dots, y_K\}$ per medical user prompt.
2. Evaluating each completion with verifiable rule-based reward functions:
   - **Safety Reward ($r_{safety}$)**: Penalizes doctor persona claims ($-1.0$), missing emergency escalation on life-threatening prompts ($-2.0$), and unverified child/pediatric dosing ($-2.0$).
   - **Style Reward ($r_{style}$)**: Penalizes marketplace lead-generation boilerplate ($-1.0$ for "ChatDoctor", "consult me", etc.).
   - **Length Reward ($r_{length}$)**: Applies penalties for verbosity exceeding target response lengths.
3. Computing group-relative advantages:
   $$A_i = \frac{R_i - \text{mean}(R)}{\text{std}(R) + \epsilon}$$
4. Updating the policy using a clipped surrogate objective with DAPO token normalization and a KL divergence penalty ($\beta = 0.02$) anchoring the model to the reference SFT weights.

### 3. Evaluation & Clinical Benchmarking
The evaluation suite (`sft/evaluate.py`) executes standard medical safety and knowledge probes:
- **Tier-0 Emergency Escalation Rate (EER)**: Checks acute clinical emergencies (acute myocardial infarction, anaphylaxis, stroke, major hemorrhages) for mandatory immediate escalation advice.
- **Clinical Knowledge Accuracy**: Multiple-choice diagnostic and pharmacology probes assessing factual correctness.
- **Persona & Style Gating**: Verifies the model never falsely claims medical credentials or solicits paid consultations.

---

## 📂 Repository Structure

```
MEDICAL-AI/
├── README.md                      # Primary project overview and documentation
├── requirements.txt               # Base dependencies
│
├── data/
│   ├── processed/
│   │   └── integrated_conversations_ner_validated.csv  # Read-only source corpus
│   └── sft/v1/
│       ├── train.jsonl            # Stratified training set
│       ├── val.jsonl              # Held-out validation set
│       └── prep_report.json       # Preprocessing and dialogue statistics
│
├── sft/                           # SFT, GRPO, and Evaluation scaffold
│   ├── README.md                  # SFT-specific guide & Kaggle instructions
│   ├── build_jsonl.py             # Preprocessing & dialogue builder
│   ├── train_sft.py               # LoRA / QLoRA SFT training script
│   ├── train_grpo.py              # GRPO RL training script
│   ├── kaggle_train.py            # End-to-end single-file script for Kaggle notebooks
│   ├── evaluate.py                # Clinical evaluation, accuracy, and reward scoring
│   ├── build_manifest.py          # Builds protected-artifact SHA-256 manifest
│   ├── verify_manifest.py         # Verifies repository file integrity
│   ├── protected_manifest.json    # SHA-256 integrity hashes
│   ├── requirements-cpu.txt       # Minimal CPU dependencies
│   ├── tokenizer_snapshot/        # Offline Mistral-7B-Instruct-v0.3 tokenizer
│   └── tests/
│       └── test_smoke.py          # Unit & tokenizer smoke tests (22 tests)
│
├── notebooks/                     # Exploratory analysis & data sanitization
│   ├── 01_eda.ipynb               # Initial Exploratory Data Analysis
│   ├── 02_edamedquad.ipynb        # MedQuAD analysis
│   ├── 03_audit_sanitize_validate.ipynb # Audit and sanitization
│   ├── 04_scispacy_ner.ipynb      # SciSpacy NER processing
│   ├── 05_ner_validation.ipynb    # NER validation
│   └── 06_cleaning_pipeline.ipynb # Cleaning pipeline
│
└── docs/                          # Detailed architecture specifications
    ├── ARCHITECTURE.md            # Approved architecture & design decisions
    ├── FINETUNE_RL.md             # SFT and RL alignment strategy
    ├── DATAPIPE.md                # Data processing specifications
    ├── PROJECT.md                 # Project lifecycle and milestones
    └── IMPLEMENTATION_PLAN.md     # Implementation phases
```

---

## 🚀 Quickstart & Execution Guide

### Step 1: Environment Setup (Local CPU)
```bash
# Clone and enter the repository
cd MEDICAL-AI

# Create and activate virtual environment
python -m venv sft/.venv
.\sft\.venv\Scripts\Activate.ps1    # On Windows PowerShell
# source sft/.venv/bin/activate     # On Linux / macOS

# Install lightweight CPU verification requirements
pip install -r sft/requirements-cpu.txt
```

### Step 2: Run Smoke Tests & Manifest Verification
```bash
# Run unit tests (offline tokenizer, masking logic, verifier)
python -m unittest discover -v -s sft/tests

# Verify repository integrity
python sft/verify_manifest.py
```

### Step 3: Data Preprocessing
```bash
# Convert raw integrated CSV into clean SFT/GRPO JSONL datasets
python sft/build_jsonl.py
```
This produces `data/sft/v1/train.jsonl` and `data/sft/v1/val.jsonl`.

### Step 4: Training on GPU (Kaggle T4 or A100)

#### Option A: Single-File Notebook Execution
Upload `train.jsonl` and `val.jsonl` to Kaggle as a private dataset. Create a new notebook with GPU accelerator (T4 x2 or P100), paste the contents of `sft/kaggle_train.py`, and run.

#### Option B: Step-by-Step Training

**1. Run SFT:**
```bash
python sft/train_sft.py \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.3 \
    --train_data data/sft/v1/train.jsonl \
    --val_data   data/sft/v1/val.jsonl   \
    --output_dir runs/sft-mistral-lora   \
    --use_qlora                          \
    --max_len 2048                       \
    --batch_size 1                       \
    --grad_accum 32
```

**2. Run GRPO Alignment:**
```bash
python sft/train_grpo.py \
    --model_path runs/sft-mistral-lora/final \
    --train_data data/sft/v1/train.jsonl     \
    --output_dir runs/grpo-mistral           \
    --num_gen 4                              \
    --max_steps 300                          \
    --beta 0.02                              \
    --use_qlora
```

### Step 5: Evaluate Model Accuracy & Safety
Run the clinical evaluation harness on your trained checkpoints:

```bash
# Evaluate SFT Checkpoint
python sft/evaluate.py \
    --base_model mistralai/Mistral-7B-Instruct-v0.3 \
    --adapter_path runs/sft-mistral-lora/final \
    --use_qlora \
    --output_report runs/sft_eval.json

# Evaluate GRPO Aligned Checkpoint
python sft/evaluate.py \
    --base_model mistralai/Mistral-7B-Instruct-v0.3 \
    --adapter_path runs/grpo-mistral/final \
    --use_qlora \
    --output_report runs/grpo_eval.json
```

Sample Scorecard Output:
```
=================================================================
             MEDICAL-AI EVALUATION SCORECARD
=================================================================
Total Probes Evaluated:         12
Emergency Escalation Rate (EER): 100.0%  [PASS >=90%]
Clinical MCQ Accuracy:          100.0%  [PASS >=75%]
Persona Violation Rate:         0.0%    [PASS <=1%]
Lead-Gen Violation Rate:        0.0%    [PASS <=1%]
Pediatric Dosing Safety:        100.0%  [PASS 100%]
-----------------------------------------------------------------
GRPO Reward Metrics (Higher is better):
  Avg Safety Reward:            0.0
  Avg Style Reward:             0.0
  Avg Length Reward:            0.0
  Avg Composite Reward:         0.0
-----------------------------------------------------------------
OVERALL STATUS: ALL GATES PASSED (ACCEPT)
=================================================================
```

---

## 📊 Summary of Architectural Differences

| Feature | Mistral-7B-Instruct-v0.3 (Current) | Qwen3-1.7B / 8B (Former Friend's Baseline) |
|---|---|---|
| **License** | Apache-2.0 | Apache-2.0 |
| **Chat Template** | Native `[INST] ... [/INST]` | ChatML (`<\|im_start\|>`) |
| **System Injection** | Leading `[INST]` turn | Standalone system turn |
| **EOS Token** | `</s>` (id=2) | `<\|im_end\|>` (id=151645) |
| **Thinking Mode** | Disabled (not present) | Requires manual `enable_thinking=False` |
| **Vocab Size** | 32,768 tokens | 151,936 tokens |
| **Post-Training** | SFT $\rightarrow$ GRPO with rule rewards | SFT only |
| **VRAM on T4** | ~6.5 GB with 4-bit QLoRA | ~7.3 GB with fp16 LoRA |

---

## 📜 Ethical & Compliance Notes
1. **MedDialog Slice**: Used under academic research terms. Not for commercial distribution.
2. **HealthChat**: Derived from LMSYS / WildChat non-toxic health dialogues.
3. **Data Privacy**: No patient identifiable information (PII) is included in the training splits.
