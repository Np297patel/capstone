"""
sft/train_grpo.py — GRPO (RL) training for Mistral-7B-Instruct-v0.3

Runs AFTER SFT. Takes the SFT adapter checkpoint as the starting policy.
Uses rule-based reward functions only — no external judge, no RAG.

Usage:
    python sft/train_grpo.py \
        --model_path  runs/sft-mistral-lora/final \
        --train_data  data/sft/v1/train.jsonl      \
        --output_dir  runs/grpo-mistral            \
        --max_steps   500

GRPO overview:
    - For each prompt, sample K=8 completions from the current policy.
    - Score each completion with composite reward R = R_safety + R_style + R_length.
    - Compute group-relative advantages: A_i = (R_i - mean(R)) / std(R).
    - Update policy with clipped PPO-style objective (loss_type="dapo" in TRL).
    - A KL penalty (beta) anchors the policy to the reference (SFT model).

Reward functions (no judge model needed):
    r_safety  — penalises persona claims, missing escalation for tier-0 prompts,
                child-dose mentions
    r_style   — penalises lead-gen / sign-off language
    r_length  — penalises verbosity beyond a per-prompt target

Data format — same JSONL as SFT:
    {"messages": [{"role": "user", "content": "..."}, ...]}
    Optionally include "tier0": true for emergency prompts and
    "target_len": 200 for length targets (defaults to 200 tokens).
"""

import argparse
import re

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

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

# ── regex patterns ─────────────────────────────────────────────────────────────

# Doctor persona / credential claims
PERSONA_RX = re.compile(
    r"\b(I am (a |your )?(doctor|physician|Dr\.?|specialist)|"
    r"as (a |your )?(doctor|physician|clinician)|"
    r"my (clinic|practice|profile)|"
    r"contact me|consult me|video consultation)\b",
    re.IGNORECASE,
)

# Lead-gen / sign-off boilerplate (common in MedDialog)
LEADGEN_RX = re.compile(
    r"(hope (this|i have) (helps|answered)|"
    r"feel free to (consult|contact)|"
    r"chat doctor|chatdoctor|"
    r"hope I have answered your query)",
    re.IGNORECASE,
)

# Emergency escalation language (what we WANT for tier-0 prompts)
ESCALATE_RX = re.compile(
    r"(emergency\s*services|call\s*(9[01][01]|112|999|000)|"
    r"emergency\s*room|go\s*to\s*(the\s*)?(er|hospital|a&e)|"
    r"call\s*an\s*ambulance|seek\s*immediate|urgent\s*care\s*now)",
    re.IGNORECASE,
)

# Child / paediatric dose mentions (flag — should not appear in model output)
DOSE_RX = re.compile(
    r"\b(\d+\s*(mg|mcg|ml|units?)/kg|"
    r"paediatric\s*dose|child\s*(dose|dosing)|"
    r"infant\s*(dose|dosing))\b",
    re.IGNORECASE,
)


# ── reward functions ───────────────────────────────────────────────────────────
# TRL GRPOTrainer calls each reward_func(prompts, completions, **extra_columns).
# completions is a list of lists: [[{"role":"assistant","content":"..."}], ...]
# Extra columns are passed as kwargs aligned with the completion list.

def _text(completion) -> str:
    """Extract the assistant reply text from a TRL completion object."""
    if isinstance(completion, list):
        return completion[-1]["content"] if completion else ""
    return str(completion)


def r_safety(prompts, completions, tier0=None, child_context=None, **kwargs):
    """
    Safety reward.
      -1.0  if persona claim present
      -2.0  if tier-0 prompt but no escalation language
      -2.0  if child/paediatric dosing mentioned in output
      Rewards are additive (can stack to -5.0 in worst case).
      0.0   if none of the above.
    """
    if tier0 is None:
        tier0 = [False] * len(completions)
    if child_context is None:
        child_context = [False] * len(completions)

    scores = []
    for comp, t0, ch in zip(completions, tier0, child_context):
        t = _text(comp)
        score = 0.0
        if PERSONA_RX.search(t):
            score -= 1.0
        if t0 and not ESCALATE_RX.search(t):
            score -= 2.0
        if ch and DOSE_RX.search(t):
            score -= 2.0
        scores.append(score)
    return scores


def r_style(prompts, completions, **kwargs):
    """
    Style reward.
      0.0  if no lead-gen or persona language.
      -1.0 if lead-gen / sign-off detected.
    """
    return [
        -1.0 if (PERSONA_RX.search(_text(c)) or LEADGEN_RX.search(_text(c))) else 0.0
        for c in completions
    ]


def r_length(prompts, completions, target_len=None, **kwargs):
    """
    Length penalty. Returns 0.0 if within target, negative if too long.
    target_len: per-prompt target in tokens (approximate via char/4).
    Penalises responses that exceed target_len by more than 20%.
    """
    if target_len is None:
        target_len = [200] * len(completions)

    scores = []
    for comp, tgt in zip(completions, target_len):
        approx_tokens = len(_text(comp)) / 4.0
        if approx_tokens <= tgt * 1.2:
            scores.append(0.0)
        else:
            overshoot = (approx_tokens - tgt) / tgt
            scores.append(-min(1.0, overshoot))
    return scores


# ── data prep ─────────────────────────────────────────────────────────────────

def prepare_grpo_dataset(raw_dataset, tokenizer):
    """
    Convert JSONL records into the format GRPOTrainer expects:
        {"prompt": [{"role": "user", "content": "..."}], ...extra columns...}

    The prompt is the conversation up to (but not including) the final assistant turn.
    Extra columns (tier0, target_len) are passed through to reward functions.
    """
    records = []
    for item in raw_dataset:
        messages = item.get("messages", [])
        if not messages:
            continue

        # Inject system prompt
        if messages[0]["role"] != "system":
            messages = [{"role": "system", "content": SYSTEM_PROMPT_V1}] + list(messages)

        # Drop trailing assistant turns — GRPO generates those
        prompt_messages = messages
        if messages[-1]["role"] == "assistant":
            prompt_messages = messages[:-1]

        # Need at least one user turn
        if not any(m["role"] == "user" for m in prompt_messages):
            continue

        records.append({
            "prompt":     prompt_messages,
            "tier0":      bool(item.get("tier0", False)),
            "target_len": int(item.get("target_len", 200)),
        })
    return records


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GRPO RL training for Mistral-7B-Instruct-v0.3")
    parser.add_argument("--model_path",  required=True,
                        help="Path to the SFT adapter or merged model (local dir).")
    parser.add_argument("--train_data",  required=True)
    parser.add_argument("--output_dir",  default="runs/grpo-mistral")
    parser.add_argument("--max_steps",   type=int, default=500)
    parser.add_argument("--num_gen",     type=int, default=8,
                        help="Number of completions per prompt (K). Reduce to 4 on T4.")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--lr",          type=float, default=5e-6)
    parser.add_argument("--beta",        type=float, default=0.02,
                        help="KL penalty coefficient (anchors policy to SFT).")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--use_qlora",   action="store_true",
                        help="Load model in 4-bit for memory-constrained GPUs.")
    args = parser.parse_args()

    print(f"\n=== GRPO: Mistral-7B-Instruct-v0.3 ===")
    print(f"  model_path:    {args.model_path}")
    print(f"  max_steps:     {args.max_steps}")
    print(f"  K (num_gen):   {args.num_gen}")
    print(f"  beta (KL):     {args.beta}")
    print(f"  lr:            {args.lr}\n")

    # ── tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.padding_side = "left"  # left-pad for generation

    # ── dataset ──
    raw = load_dataset("json", data_files={"train": args.train_data})
    records = prepare_grpo_dataset(raw["train"], tokenizer)
    if not records:
        raise RuntimeError("No GRPO training records — check your JSONL format.")
    print(f"  GRPO prompts: {len(records)}")

    from datasets import Dataset
    grpo_ds = Dataset.from_list(records)

    # ── LoRA config for GRPO (smaller r — we're fine-tuning an already-fine-tuned model) ──
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── GRPO config ──
    grpo_args = GRPOConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=20,
        # Batch: each step processes batch_size prompts × num_generations completions
        per_device_train_batch_size=1,      # 1 prompt per step on T4
        gradient_accumulation_steps=4,
        num_generations=args.num_gen,       # K completions per prompt
        max_completion_length=args.max_new_tokens,
        # GRPO objective
        loss_type="dapo",                   # normalises by active tokens, removes length bias
        beta=args.beta,                     # KL anchor — small but non-zero to stay near SFT
        epsilon=0.2,
        epsilon_high=0.28,
        mask_truncated_completions=True,    # don't penalise completions cut by max_completion_length
        # Reward weights: [r_safety, r_style, r_length]
        # r_safety is the most important — give it the highest weight
        reward_weights=[2.0, 1.0, 1.0],
        # Training schedule
        max_steps=args.max_steps,
        save_steps=100,
        logging_steps=5,
        # Precision — T4: fp16; A100: switch to bf16=True, fp16=False
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        report_to="none",
        # vLLM for faster generation — set to False if not installed on Kaggle
        use_vllm=False,
    )

    # ── trainer ──
    trainer = GRPOTrainer(
        model=args.model_path,
        reward_funcs=[r_safety, r_style, r_length],
        args=grpo_args,
        train_dataset=grpo_ds,
        processing_class=tokenizer,
        peft_config=lora_cfg,
    )

    print("\nStarting GRPO training...")
    trainer.train()

    print(f"\nSaving final GRPO adapter to {args.output_dir}/final ...")
    trainer.save_model(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")
    print("Done.")


if __name__ == "__main__":
    main()
