"""
sft/build_jsonl.py — Convert integrated_conversations_ner_validated.csv
                     into train.jsonl / val.jsonl for Mistral SFT + GRPO.

Usage:
    python sft/build_jsonl.py

Output:
    data/sft/v1/train.jsonl
    data/sft/v1/val.jsonl
    data/sft/v1/prep_report.json

Each output line:
    {
        "messages": [
            {"role": "user",      "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
        ],
        "dialogue_id": "...",
        "source": "...",
        "n_exchanges": 2
    }

Split: 90% train / 10% val, split by dialogue_id (never by row).
MedDialog dialogues are always 2 turns (1 exchange).
HealthChat dialogues can be multi-turn.
"""

import json
import random
import re
from pathlib import Path

import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent
CSV_PATH   = REPO_ROOT / "data" / "processed" / "integrated_conversations_ner_validated.csv"
OUT_DIR    = REPO_ROOT / "data" / "sft" / "v1"
TRAIN_PATH = OUT_DIR / "train.jsonl"
VAL_PATH   = OUT_DIR / "val.jsonl"
REPORT_PATH= OUT_DIR / "prep_report.json"

# ── config ────────────────────────────────────────────────────────────────────
VAL_RATIO   = 0.10   # 10% of dialogues go to val
RANDOM_SEED = 42
MIN_USER_TOKENS   = 5    # approximate (chars / 4)
MIN_ASSIST_TOKENS = 20

# ── cleaning patterns ─────────────────────────────────────────────────────────

# ChatDoctor / MedDialog template boilerplate to strip from assistant turns
STRIP_PATTERNS = [
    re.compile(r"hope\s+i\s+have\s+answered\s+your\s+(query|question)\.?", re.I),
    re.compile(r"hope\s+this\s+(helps|helped|answer).*", re.I),
    re.compile(r"feel\s+free\s+to\s+(consult|contact|ask).*", re.I),
    re.compile(r"chat\s*doctor[i]?", re.I),
    re.compile(r"chatdoctor[i]?", re.I),
    re.compile(r"thanks\s+for\s+your\s+query\.?", re.I),
    re.compile(r"thanks\s+for\s+using\s+.*", re.I),
    re.compile(r"do\s+write\s+back\s+.*", re.I),
    re.compile(r"take\s+care\.?\s*$", re.I),
]

# Credential / persona claims — drop the whole dialogue if found
PERSONA_RX = re.compile(
    r"\b(I\s+am\s+(a\s+)?(doctor|physician|Dr\.?)|"
    r"as\s+(a\s+|your\s+)?(doctor|physician|clinician)|"
    r"my\s+(clinic|practice|profile)|"
    r"contact\s+me|consult\s+me|"
    r"video\s+consultation)\b",
    re.I,
)


def clean_text(text: str) -> str:
    """Strip template boilerplate from the end of assistant turns."""
    t = text.strip()
    for pat in STRIP_PATTERNS:
        t = pat.sub("", t).strip()
    # Collapse excessive whitespace
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r" {2,}", " ", t)
    return t.strip()


def is_english(text: str) -> bool:
    """Quick heuristic — if >80% of chars are ASCII it's likely English."""
    if not text:
        return False
    ascii_count = sum(1 for c in text if ord(c) < 128)
    return ascii_count / len(text) > 0.80


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading CSV from {CSV_PATH} ...")
    df = pd.read_csv(
        CSV_PATH,
        low_memory=False,
        dtype={"original_id": str, "dialogue_id": str},
    )
    # Drop null utterances
    df = df.dropna(subset=["utterance"])
    df["utterance"] = df["utterance"].astype(str).str.strip()
    df = df[df["utterance"] != ""]

    print(f"  Rows after dropping nulls: {len(df):,}")
    print(f"  Unique dialogues:          {df['dialogue_id'].nunique():,}")

    # ── build dialogues ──
    drop_reasons = {
        "empty_after_clean":    0,
        "too_short":            0,
        "persona_claim":        0,
        "bad_structure":        0,   # not starting with user or not alternating
        "not_english":          0,
        "no_full_exchange":     0,
    }

    dialogues = []

    for dlg_id, grp in df.groupby("dialogue_id", sort=False):
        grp = grp.sort_values("turn_id").reset_index(drop=True)
        source = grp["source_dataset"].iloc[0]

        # Map speaker → role
        role_map = {"user": "user", "assistant": "assistant"}
        turns = []
        for _, row in grp.iterrows():
            role = role_map.get(str(row["speaker"]).lower())
            if role is None:
                break
            text = str(row["utterance"]).strip()
            if role == "assistant":
                text = clean_text(text)
            turns.append({"role": role, "content": text})

        # Must start with user
        if not turns or turns[0]["role"] != "user":
            drop_reasons["bad_structure"] += 1
            continue

        # Cut at first structural defect (same role twice, empty turn)
        clean_turns = []
        prev_role = None
        for t in turns:
            if t["role"] == prev_role:
                break   # same role twice — cut here
            if not t["content"]:
                break
            clean_turns.append(t)
            prev_role = t["role"]

        # Must end on assistant turn (complete exchange)
        while clean_turns and clean_turns[-1]["role"] != "assistant":
            clean_turns.pop()

        # Need at least one full user+assistant exchange
        if len(clean_turns) < 2:
            drop_reasons["no_full_exchange"] += 1
            continue

        # Check for persona claims in any assistant turn
        has_persona = any(
            PERSONA_RX.search(t["content"])
            for t in clean_turns if t["role"] == "assistant"
        )
        if has_persona:
            drop_reasons["persona_claim"] += 1
            continue

        # Length check on first user and first assistant turn
        first_user    = next(t for t in clean_turns if t["role"] == "user")
        first_assist  = next(t for t in clean_turns if t["role"] == "assistant")
        if approx_tokens(first_user["content"]) < MIN_USER_TOKENS:
            drop_reasons["too_short"] += 1
            continue
        if approx_tokens(first_assist["content"]) < MIN_ASSIST_TOKENS:
            drop_reasons["too_short"] += 1
            continue

        # English check on user turns
        if not all(is_english(t["content"]) for t in clean_turns if t["role"] == "user"):
            drop_reasons["not_english"] += 1
            continue

        n_exchanges = sum(1 for t in clean_turns if t["role"] == "assistant")

        dialogues.append({
            "dialogue_id": dlg_id,
            "source":      source,
            "n_exchanges": n_exchanges,
            "messages":    clean_turns,
        })

    total_kept = len(dialogues)
    total_dropped = sum(drop_reasons.values())
    print(f"\n  Kept:    {total_kept:,} dialogues")
    print(f"  Dropped: {total_dropped:,} dialogues")
    for reason, count in drop_reasons.items():
        if count:
            print(f"    {reason}: {count:,}")

    # ── split by dialogue_id (90/10) ──
    random.seed(RANDOM_SEED)
    all_ids = [d["dialogue_id"] for d in dialogues]
    random.shuffle(all_ids)
    n_val = max(1, int(len(all_ids) * VAL_RATIO))
    val_ids  = set(all_ids[:n_val])
    train_ids = set(all_ids[n_val:])

    train_dialogues = [d for d in dialogues if d["dialogue_id"] in train_ids]
    val_dialogues   = [d for d in dialogues if d["dialogue_id"] in val_ids]

    print(f"\n  Train: {len(train_dialogues):,} dialogues")
    print(f"  Val:   {len(val_dialogues):,} dialogues")

    # Source breakdown
    for split_name, split_data in [("train", train_dialogues), ("val", val_dialogues)]:
        from collections import Counter
        src_counts = Counter(d["source"] for d in split_data)
        multi = sum(1 for d in split_data if d["n_exchanges"] > 1)
        print(f"\n  {split_name} sources: {dict(src_counts)}")
        print(f"  {split_name} multi-turn dialogues: {multi:,}")

    # ── write JSONL ──
    def write_jsonl(path, data):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                # Only write the fields the training script needs
                f.write(json.dumps({
                    "messages":    item["messages"],
                    "dialogue_id": item["dialogue_id"],
                    "source":      item["source"],
                    "n_exchanges": item["n_exchanges"],
                }, ensure_ascii=False) + "\n")
        print(f"  Written: {path}  ({len(data):,} lines)")

    write_jsonl(TRAIN_PATH, train_dialogues)
    write_jsonl(VAL_PATH,   val_dialogues)

    # ── prep report ──
    from collections import Counter
    report = {
        "total_dialogues_in_csv":  df["dialogue_id"].nunique(),
        "total_rows_in_csv":       len(df),
        "kept":                    total_kept,
        "dropped":                 total_dropped,
        "drop_reasons":            drop_reasons,
        "train_dialogues":         len(train_dialogues),
        "val_dialogues":           len(val_dialogues),
        "train_sources":           dict(Counter(d["source"] for d in train_dialogues)),
        "val_sources":             dict(Counter(d["source"] for d in val_dialogues)),
        "train_multi_turn":        sum(1 for d in train_dialogues if d["n_exchanges"] > 1),
        "val_multi_turn":          sum(1 for d in val_dialogues   if d["n_exchanges"] > 1),
        "split_seed":              RANDOM_SEED,
        "val_ratio":               VAL_RATIO,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Report:  {REPORT_PATH}")

    print("\nDone. Next step: upload train.jsonl + val.jsonl to Kaggle as a private dataset.")


if __name__ == "__main__":
    main()
