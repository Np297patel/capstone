"""
sft/train_sft.py — LoRA / QLoRA SFT for Mistral-7B-Instruct-v0.3

Usage (Kaggle T4 — QLoRA recommended):
    python sft/train_sft.py \
        --model_name_or_path mistralai/Mistral-7B-Instruct-v0.3 \
        --train_data data/sft/v1/train.jsonl \
        --val_data   data/sft/v1/val.jsonl   \
        --output_dir runs/sft-mistral-lora   \
        --use_qlora                          \
        --max_len 2048

Usage (A100 / full bf16 LoRA):
    python sft/train_sft.py \
        --model_name_or_path mistralai/Mistral-7B-Instruct-v0.3 \
        --train_data data/sft/v1/train.jsonl \
        --val_data   data/sft/v1/val.jsonl   \
        --output_dir runs/sft-mistral-lora   \
        --max_len 2048

Data format — each JSONL line:
    {"messages": [
        {"role": "system",    "content": "You are a health information assistant..."},
        {"role": "user",      "content": "..."},
        {"role": "assistant", "content": "..."},
        ...
    ]}

The system role is optional; if absent, the built-in SYSTEM_PROMPT_V1 is prepended.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# ── constants ─────────────────────────────────────────────────────────────────

MODEL_REVISION = "e0bc86c23ce5aae1db576c8cca6f06f1f73af2db"

SYSTEM_PROMPT_V1 = (
    "You are a general health-information assistant. "
    "You are an AI, not a doctor: you cannot examine the user, diagnose, or prescribe. "
    "Use everything the user has told you in this conversation. "
    "If key details are missing, ask one or two focused questions. "
    "State uncertainty plainly. "
    "If there are signs of an emergency, tell the user to contact local emergency services now. "
    "Recommend seeing a qualified clinician for severe, persistent, or worsening symptoms, "
    "and for any prescription medicine or dosing decision."
)

# Mistral [INST] control tokens (v0.3 has these as single tokens)
INST_OPEN  = "[INST]"
INST_CLOSE = "[/INST]"


# ── tokenization + label masking ──────────────────────────────────────────────

def build_masked_example(messages: list[dict], tokenizer, max_len: int) -> dict | None:
    """
    Render a conversation in Mistral format and build the assistant-only label mask.

    Mistral template (v0.3):
        <s>[INST] {system}

    {user_1} [/INST] {assistant_1}</s>[INST] {user_2} [/INST] {assistant_2}</s>...

    We supervise ONLY the assistant tokens (content + </s>).
    Everything else — <s>, [INST], user turns, [/INST] — gets label=-100.

    Returns None if the sequence exceeds max_len (dropped, not truncated mid-turn).
    """
    tok = tokenizer

    # Inject system prompt if missing
    if not messages or messages[0]["role"] != "system":
        messages = [{"role": "system", "content": SYSTEM_PROMPT_V1}] + list(messages)

    # Render the full string with the tokenizer's built-in chat template
    full_text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    input_ids = tok.encode(full_text, add_special_tokens=False)

    if len(input_ids) > max_len:
        return None

    labels = [-100] * len(input_ids)

    # Find the [/INST] token id — in v0.3 it's a single token
    inst_close_ids = tok.encode(INST_CLOSE, add_special_tokens=False)
    eos_id = tok.eos_token_id  # </s> = 2

    if len(inst_close_ids) != 1:
        # Fallback: tokenizer doesn't have [/INST] as a single token.
        # This shouldn't happen with Mistral-7B-Instruct-v0.3 but guard anyway.
        raise RuntimeError(
            f"[/INST] tokenizes to {len(inst_close_ids)} tokens (expected 1). "
            "Check that the tokenizer snapshot is Mistral-7B-Instruct-v0.3."
        )
    inst_close_id = inst_close_ids[0]

    # Scan: after each [/INST], label tokens as supervised until </s> (inclusive)
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

    # Require at least one supervised token
    if all(l == -100 for l in labels):
        return None

    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


def tokenize_dataset(dataset, tokenizer, max_len: int, split_name: str = ""):
    results = []
    dropped = 0
    for item in dataset:
        messages = item.get("messages", [])
        if not messages:
            dropped += 1
            continue
        ex = build_masked_example(messages, tokenizer, max_len)
        if ex is None:
            dropped += 1
        else:
            results.append(ex)
    print(f"  [{split_name}] kept={len(results)}, dropped={dropped}")
    return results


# ── model + tokenizer setup ───────────────────────────────────────────────────

def load_tokenizer(model_name: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(model_name, revision=MODEL_REVISION)
    # Mistral-7B-Instruct-v0.3 may not have pad_token set
    if tok.pad_token_id is None:
        # Use unk_token as pad — DO NOT use eos_token (corrupts loss mask)
        tok.pad_token = tok.unk_token
        tok.pad_token_id = tok.unk_token_id
    tok.padding_side = "right"  # required for causal LM training
    return tok


def load_model(model_name: str, use_qlora: bool) -> AutoModelForCausalLM:
    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,  # T4 has no bf16
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=MODEL_REVISION,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=False,
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )
    else:
        # Full fp16 LoRA — needs ~16 GB VRAM (fits A100/3090/4090)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=MODEL_REVISION,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=False,
        )
    return model


def attach_lora(model, use_qlora: bool) -> None:
    """Apply LoRA config in-place using PEFT."""
    lora_cfg = LoraConfig(
        r=16 if use_qlora else 64,           # r=16 for QLoRA (< 40 GB), r=64 for full
        lora_alpha=32 if use_qlora else 128,
        lora_dropout=0.05,
        target_modules="all-linear",         # all attention + MLP projections
        bias="none",
        task_type="CAUSAL_LM",
    )
    # get_peft_model wraps the model in-place; no need to reassign
    peft_model = get_peft_model(model, lora_cfg)
    peft_model.print_trainable_parameters()
    return peft_model


# ── training ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SFT for Mistral-7B-Instruct-v0.3")
    parser.add_argument("--model_name_or_path", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--train_data",  required=True)
    parser.add_argument("--val_data",    required=True)
    parser.add_argument("--output_dir",  default="runs/sft-mistral-lora")
    parser.add_argument("--max_len",     type=int, default=2048)
    parser.add_argument("--epochs",      type=int, default=1)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--batch_size",  type=int, default=1,
                        help="Per-device train batch size. Use 1 for T4 QLoRA.")
    parser.add_argument("--grad_accum",  type=int, default=32,
                        help="Gradient accumulation steps. Effective batch = batch_size * grad_accum.")
    parser.add_argument("--use_qlora",   action="store_true",
                        help="Use 4-bit QLoRA (NF4). Recommended for T4/P100 (16 GB).")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    print(f"\n=== SFT: Mistral-7B-Instruct-v0.3 ===")
    print(f"  QLoRA:        {args.use_qlora}")
    print(f"  max_len:      {args.max_len}")
    print(f"  effective_bs: {args.batch_size * args.grad_accum}")
    print(f"  epochs:       {args.epochs}")
    print(f"  lr:           {args.lr}\n")

    # ── tokenizer ──
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.model_name_or_path)

    # ── raw datasets ──
    print("Loading raw JSONL datasets...")
    raw = load_dataset(
        "json",
        data_files={"train": args.train_data, "validation": args.val_data},
    )

    # ── tokenize + mask ──
    print("Tokenizing and masking (assistant-only labels)...")
    train_examples = tokenize_dataset(raw["train"], tokenizer, args.max_len, "train")
    val_examples   = tokenize_dataset(raw["validation"], tokenizer, args.max_len, "val")

    if not train_examples:
        raise RuntimeError("No training examples after tokenization — check your JSONL format.")

    # Wrap as HF Dataset (simple list-of-dicts)
    from datasets import Dataset
    train_ds = Dataset.from_list(train_examples)
    val_ds   = Dataset.from_list(val_examples)

    # ── model ──
    print("Loading model...")
    model = load_model(args.model_name_or_path, args.use_qlora)
    model = attach_lora(model, args.use_qlora)

    # ── collator ──
    # DataCollatorForSeq2Seq pads to the longest sequence in the batch and
    # correctly pads labels with -100 so padding doesn't contribute to loss.
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    # ── training args ──
    # T4 has no native bfloat16 — use fp16.
    # On A100/3090 you can switch to bf16=True, fp16=False.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        seed=args.seed,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        max_grad_norm=1.0,
        fp16=True,                  # T4 / P100: fp16. Switch to bf16=True on Ampere+.
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Evaluation + checkpointing
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Logging
        logging_steps=10,
        report_to="none",           # change to "wandb" if you have W&B set up
        # Misc
        dataloader_num_workers=0,   # 0 is safer on Kaggle
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,
    )

    print("\nStarting training...")
    trainer.train()

    print(f"\nSaving final adapter to {args.output_dir}/final ...")
    trainer.save_model(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")
    print("Done.")


if __name__ == "__main__":
    main()
