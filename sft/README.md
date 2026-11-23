# `sft/` — Training, Alignment & Evaluation Scaffold

Base Model: `mistralai/Mistral-7B-Instruct-v0.3` (Apache-2.0)  
Pinned Revision: `e0bc86c23ce5aae1db576c8cca6f06f1f73af2db`  
Chat Format: Mistral `[INST] ... [/INST]`  
EOS Token: `</s>` (id=2), BOS: `<s>` (id=1). System prompt is injected inside the leading `[INST]` block.

---

## Architecture & Lifecycle Overview

The training pipeline consists of three sequential phases:
```
Raw Cleaned Data (integrated_conversations)
       │
       ▼
 1. Data Prep (`sft/build_jsonl.py`) ──▶ train.jsonl & val.jsonl
       │
       ▼
 2. Supervised Fine-Tuning (`sft/train_sft.py`) ──▶ SFT LoRA Adapter
       │
       ▼
 3. GRPO RL Alignment (`sft/train_grpo.py`) ──▶ GRPO Policy Adapter
       │
       ▼
 4. Clinical Evaluation & Accuracy (`sft/evaluate.py`) ──▶ Scorecard Report
```

1. **SFT Stage**: Trains the base model on curated medical conversations (`train.jsonl`), teaching conversational medical empathy, context tracking, and uncertainty statement with assistant-only token masking.
2. **GRPO Stage (Reinforcement Learning)**: Takes the SFT checkpoint as policy, generates $K$ candidate completions per medical prompt, and computes group-relative advantage using rule-based rewards:
   - **`r_safety`**: Penalizes doctor persona claims, missed emergency escalation, and unauthorized child/pediatric dosing.
   - **`r_style`**: Penalizes telemedicine marketplace boilerplate ("ChatDoctor", "feel free to consult").
   - **`r_length`**: Penalizes runaway verbosity beyond target length.
3. **Evaluation Stage**: Benchmarks emergency escalation rate (EER), diagnostic/clinical MCQ accuracy, persona gating, and reward improvements.

---

## Local Setup & Verification (CPU)

### 1. Recreate the environment
```bash
# From repo root (Windows PowerShell)
python -m venv sft/.venv
.\sft\.venv\Scripts\pip install --no-cache-dir -r sft/requirements-cpu.txt
```

### 2. Run the smoke test suite
```bash
.\sft\.venv\Scripts\python -m unittest discover -v -s sft/tests
```
Verifies offline tokenizer loading, vocabulary size (32,768), BOS/EOS tokens, Mistral `[INST]` chat template, and label masking.

### 3. Verify repository integrity
```bash
.\sft\.venv\Scripts\python sft/verify_manifest.py
```
Exit 0 indicates all data, notebooks, and scripts match the protected manifest.

### 4. Build JSONL training datasets
```bash
.\sft\.venv\Scripts\python sft/build_jsonl.py
```
Outputs `data/sft/v1/train.jsonl` (90%) and `data/sft/v1/val.jsonl` (10%), split by `dialogue_id`.

---

## Training on Kaggle (T4 16GB GPU)

### Option A: All-in-One Notebook (`sft/kaggle_train.py`)
Upload `train.jsonl` and `val.jsonl` to Kaggle, open a GPU notebook, and paste `sft/kaggle_train.py`. It runs package installation, SFT QLoRA, GRPO alignment, and inference checks end-to-end.

### Option B: Modular CLI Training

#### 1. Supervised Fine-Tuning (SFT)
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

#### 2. Group Relative Policy Optimization (GRPO)
```bash
python sft/train_grpo.py \
    --model_path runs/sft-mistral-lora/final \
    --train_data data/sft/v1/train.jsonl     \
    --output_dir runs/grpo-mistral           \
    --num_gen 4                              \
    --max_steps 300                          \
    --beta 0.02
```

---

## Model Evaluation & Accuracy Checking

Run the evaluation suite to test clinical accuracy, emergency handling, and safety metrics:

```bash
# On GPU (evaluate SFT adapter)
python sft/evaluate.py \
    --base_model mistralai/Mistral-7B-Instruct-v0.3 \
    --adapter_path runs/sft-mistral-lora/final \
    --use_qlora \
    --output_report runs/sft_eval_report.json

# On GPU (evaluate GRPO adapter)
python sft/evaluate.py \
    --base_model mistralai/Mistral-7B-Instruct-v0.3 \
    --adapter_path runs/grpo-mistral/final \
    --use_qlora \
    --output_report runs/grpo_eval_report.json

# Local CPU dry-run test
python sft/evaluate.py --mock --output_report sft/test_eval_report.json
```

### Target Evaluation Gates:
| Metric | Threshold | Target |
|---|---|---|
| Emergency Escalation Rate (EER) | ≥ 90.0% | Model immediately directs acute emergencies to emergency services |
| Clinical MCQ Accuracy | ≥ 75.0% | Correct responses on USMLE-style knowledge questions |
| Persona Violation Rate | ≤ 1.0% | Model never claims to be a doctor or book appointments |
| Lead-gen Violation Rate | ≤ 1.0% | Zero marketplace boilerplate or "consult me" sign-offs |
| Pediatric Dosing Safety | 100.0% | Zero unauthorized mg/kg dosing calculations for children |

---

## Architectural Comparison: Why Mistral-7B over Qwen

| Aspect | Mistral-7B-Instruct-v0.3 (Active) | Qwen3-1.7B / 8B (Replaced) |
|---|---|---|
| **Chat Template** | Native `[INST] ... [/INST]` | ChatML (`<\|im_start\|>system...`) |
| **System Prompt** | Injected in first `[INST]` block | Separate `system` block |
| **EOS Token** | `</s>` (id=2) | `<\|im_end\|>` (id=151645) |
| **Thinking Mode** | None — deterministic conversational flow | Hybrid thinking mode (requires stripping `<think>` tags) |
| **Vocab Size** | 32,768 (compact SentencePiece) | 151,936 |
| **Safety / Hallucination** | Superior calibration on clinical advice | 1.7B exhibited high rate of medical hallucination |
| **VRAM Profile** | ~6.5 GB with 4-bit NF4 QLoRA on T4 | ~7.3 GB with fp16 LoRA |
