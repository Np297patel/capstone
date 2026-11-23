"""
sft/kaggle_train.py — Single-file Kaggle notebook script
Copy-paste this into a Kaggle notebook (Python script mode) or run cell by cell.

Hardware: Kaggle T4 (16 GB VRAM, fp16 only — no bf16)
Strategy:  QLoRA (4-bit NF4) for SFT, then GRPO with LoRA r=16

Assumes:
  - Your dataset is uploaded as a private Kaggle dataset at /kaggle/input/medchat-sft/
  - Files: train.jsonl, val.jsonl
  - Each line: {"messages": [{"role": "user", "content": "..."}, ...]}
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1 — Install packages
# ═══════════════════════════════════════════════════════════════════════════════

import subprocess, sys

pkgs = [
    "transformers==5.20.2",
    "peft==0.14.0",
    "trl==0.12.2",
    "accelerate==1.6.0",
    "bitsandbytes==0.46.0",
    "datasets==3.6.0",
]
subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=True)
print("Packages installed.")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2 — Config (edit these)
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_NAME   = "mistralai/Mistral-7B-Instruct-v0.3"
REVISION     = "e0bc86c23ce5aae1db576c8cca6f06f1f73af2db"
TRAIN_JSONL  = "/kaggle/input/medchat-sft/train.jsonl"
VAL_JSONL    = "/kaggle/input/medchat-sft/val.jsonl"
SFT_OUT      = "/kaggle/working/sft-mistral"
GRPO_OUT     = "/kaggle/working/grpo-mistral"
MAX_LEN      = 2048
SFT_EPOCHS   = 1
GRPO_STEPS   = 300        # reduce if running out of time (Kaggle sessions = 12h)
SEED         = 42


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3 — Imports
# ═══════════════════════════════════════════════════════════════════════════════

import re, torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, DataCollatorForSeq2Seq,
    Trainer, TrainingArguments,
)
from trl import GRPOConfig, GRPOTrainer

print(f"PyTorch:  {torch.__version__}")
print(f"CUDA:     {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:      {torch.cuda.get_device_name(0)}")
    print(f"VRAM:     {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4 — System prompt + regex patterns
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are a general health-information assistant. "
    "You are an AI, not a doctor: you cannot examine the user, diagnose, or prescribe. "
    "Use everything the user has told you in this conversation. "
    "If key details are missing, ask one or two focused questions. "
    "State uncertainty plainly. "
    "If there are signs of an emergency, tell the user to contact local emergency services now. "
    "Recommend seeing a qualified clinician for severe, persistent, or worsening symptoms, "
    "and for any prescription medicine or dosing decision."
)

PERSONA_RX  = re.compile(
    r"\b(I am (a |your )?(doctor|physician|Dr\.?)|as (a |your )?(doctor|physician)|"
    r"my (clinic|practice|profile)|contact me|consult me)\b", re.IGNORECASE)
LEADGEN_RX  = re.compile(
    r"(hope (this|i have) (helps|answered)|feel free to (consult|contact)|"
    r"chat doctor|chatdoctor)", re.IGNORECASE)
ESCALATE_RX = re.compile(
    r"(emergency\s*services|call\s*(9[01][01]|112|999)|emergency\s*room|"
    r"go\s*to\s*(the\s*)?(er|hospital|a&e)|call\s*an\s*ambulance|seek\s*immediate)",
    re.IGNORECASE)


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5 — Tokenizer
# ═══════════════════════════════════════════════════════════════════════════════

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=REVISION)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token    = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id
tokenizer.padding_side = "right"

# Quick sanity check — [/INST] must be a single token in v0.3
inst_ids = tokenizer.encode("[/INST]", add_special_tokens=False)
assert len(inst_ids) == 1, f"[/INST] is not a single token ({inst_ids}). Wrong tokenizer?"
print(f"Tokenizer ok. vocab={len(tokenizer)}, [/INST]={inst_ids[0]}, EOS={tokenizer.eos_token_id}")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6 — SFT data: build masked examples
# ═══════════════════════════════════════════════════════════════════════════════

def build_masked_example(messages, max_len):
    """Render + assistant-only label mask for Mistral [INST]...[/INST] format."""
    if not messages or messages[0]["role"] != "system":
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(messages)

    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    input_ids = tokenizer.encode(full_text, add_special_tokens=False)
    if len(input_ids) > max_len:
        return None

    labels = [-100] * len(input_ids)
    inst_close_id = tokenizer.encode("[/INST]", add_special_tokens=False)[0]
    eos_id = tokenizer.eos_token_id

    i = 0
    while i < len(input_ids):
        if input_ids[i] == inst_close_id:
            j = i + 1
            while j < len(input_ids):
                labels[j] = input_ids[j]
                if input_ids[j] == eos_id:
                    break
                j += 1
            i = j + 1
        else:
            i += 1

    if all(l == -100 for l in labels):
        return None
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


raw = load_dataset("json", data_files={"train": TRAIN_JSONL, "validation": VAL_JSONL})

def tok_split(split, name):
    out, drop = [], 0
    for item in split:
        ex = build_masked_example(item.get("messages", []), MAX_LEN)
        if ex: out.append(ex)
        else:  drop += 1
    print(f"  {name}: kept={len(out)}, dropped={drop}")
    return Dataset.from_list(out)

train_ds = tok_split(raw["train"],      "train")
val_ds   = tok_split(raw["validation"], "val")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 7 — Load model (QLoRA for T4)
# ═══════════════════════════════════════════════════════════════════════════════

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,   # T4: no bf16
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, revision=REVISION,
    quantization_config=bnb, device_map="auto",
)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_sft = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules="all-linear", bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_sft)
model.print_trainable_parameters()


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 8 — SFT training
# ═══════════════════════════════════════════════════════════════════════════════

collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, model=model,
    padding=True, pad_to_multiple_of=8, label_pad_token_id=-100,
)

sft_args = TrainingArguments(
    output_dir=SFT_OUT, seed=SEED,
    num_train_epochs=SFT_EPOCHS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,   # effective batch = 32
    learning_rate=1e-4,
    lr_scheduler_type="cosine", warmup_ratio=0.03,
    weight_decay=0.0, max_grad_norm=1.0,
    fp16=True, bf16=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    eval_strategy="steps", eval_steps=100,
    save_strategy="steps", save_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=10, report_to="none",
    dataloader_num_workers=0,
    remove_unused_columns=False,
)

sft_trainer = Trainer(
    model=model, args=sft_args,
    train_dataset=train_ds, eval_dataset=val_ds,
    data_collator=collator, processing_class=tokenizer,
)

print("Starting SFT...")
sft_trainer.train()
sft_trainer.save_model(f"{SFT_OUT}/final")
tokenizer.save_pretrained(f"{SFT_OUT}/final")
print("SFT done.")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 9 — GRPO reward functions
# ═══════════════════════════════════════════════════════════════════════════════

def _text(c):
    return c[-1]["content"] if isinstance(c, list) else str(c)

def r_safety(prompts, completions, tier0=None, **kw):
    if tier0 is None: tier0 = [False] * len(completions)
    out = []
    for comp, t0 in zip(completions, tier0):
        t, s = _text(comp), 0.0
        if PERSONA_RX.search(t):  s -= 1.0
        if t0 and not ESCALATE_RX.search(t): s -= 2.0
        out.append(s)
    return out

def r_style(prompts, completions, **kw):
    return [-1.0 if (PERSONA_RX.search(_text(c)) or LEADGEN_RX.search(_text(c))) else 0.0
            for c in completions]

def r_length(prompts, completions, target_len=None, **kw):
    if target_len is None: target_len = [200] * len(completions)
    out = []
    for comp, tgt in zip(completions, target_len):
        approx = len(_text(comp)) / 4.0
        out.append(0.0 if approx <= tgt * 1.2 else -min(1.0, (approx - tgt) / tgt))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 10 — Prepare GRPO dataset
# ═══════════════════════════════════════════════════════════════════════════════

# Reload tokenizer with left padding for generation
tokenizer.padding_side = "left"

grpo_records = []
for item in raw["train"]:
    msgs = item.get("messages", [])
    if not msgs: continue
    if msgs[0]["role"] != "system":
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + list(msgs)
    prompt = msgs[:-1] if msgs[-1]["role"] == "assistant" else msgs
    if not any(m["role"] == "user" for m in prompt): continue
    grpo_records.append({
        "prompt":     prompt,
        "tier0":      bool(item.get("tier0", False)),
        "target_len": int(item.get("target_len", 200)),
    })

grpo_ds = Dataset.from_list(grpo_records)
print(f"GRPO prompts: {len(grpo_ds)}")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 11 — GRPO training
# ═══════════════════════════════════════════════════════════════════════════════

lora_grpo = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules="all-linear", bias="none", task_type="CAUSAL_LM",
)

grpo_args = GRPOConfig(
    output_dir=GRPO_OUT, seed=SEED,
    learning_rate=5e-6,
    lr_scheduler_type="constant_with_warmup", warmup_steps=20,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=4,                  # K=4 on T4 (K=8 may OOM)
    max_completion_length=512,
    loss_type="dapo",
    beta=0.02,                          # KL anchor
    epsilon=0.2, epsilon_high=0.28,
    mask_truncated_completions=True,
    reward_weights=[2.0, 1.0, 1.0],    # safety, style, length
    max_steps=GRPO_STEPS,
    save_steps=100, logging_steps=5,
    fp16=True, bf16=False,
    gradient_checkpointing=True,
    report_to="none",
    use_vllm=False,                     # set True only if vllm is installed
)

grpo_trainer = GRPOTrainer(
    model=f"{SFT_OUT}/final",
    reward_funcs=[r_safety, r_style, r_length],
    args=grpo_args,
    train_dataset=grpo_ds,
    processing_class=tokenizer,
    peft_config=lora_grpo,
)

print("Starting GRPO...")
grpo_trainer.train()
grpo_trainer.save_model(f"{GRPO_OUT}/final")
tokenizer.save_pretrained(f"{GRPO_OUT}/final")
print("GRPO done.")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 12 — Quick inference check
# ═══════════════════════════════════════════════════════════════════════════════

from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, revision=REVISION,
    quantization_config=bnb, device_map="auto",
)
final_model = PeftModel.from_pretrained(base, f"{GRPO_OUT}/final")
final_model.eval()

test_messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user",   "content": "I have had a sore throat and mild fever for two days. What should I do?"},
]
prompt_text = tokenizer.apply_chat_template(
    test_messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(prompt_text, return_tensors="pt").to(final_model.device)
with torch.no_grad():
    out = final_model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.4, top_p=0.9, repetition_penalty=1.05,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("\n=== Model reply ===")
print(reply)
