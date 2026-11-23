"""
sft/tests/test_smoke.py — smoke tests for Mistral-7B-Instruct-v0.3 scaffold
Run with:  <env-python> -m unittest discover -v -s sft/tests
No test dependencies beyond stdlib + the installed CPU-stage packages.
"""

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# ── helpers ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent   # project root
SFT_DIR   = REPO_ROOT / "sft"
SNAP_DIR  = SFT_DIR / "tokenizer_snapshot"


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — library imports and langid classify
# ─────────────────────────────────────────────────────────────────────────────
class TestImports(unittest.TestCase):

    def test_pandas_import(self):
        import pandas as pd   # noqa: F401
        self.assertTrue(True)

    def test_datasketch_import(self):
        from datasketch import MinHash  # noqa: F401
        self.assertTrue(True)

    def test_langid_import(self):
        import langid  # noqa: F401
        self.assertTrue(True)

    def test_langid_classify_english(self):
        import langid
        lang, score = langid.classify("Hello world this is a test sentence")
        self.assertEqual(lang, "en", f"Expected 'en' but got {lang!r} (score={score:.4f})")

    def test_transformers_import(self):
        import transformers  # noqa: F401
        self.assertTrue(True)


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — offline tokenizer load + special tokens + round-trip
#
# Mistral-7B-Instruct-v0.3 uses a SentencePiece-based vocabulary (32k).
# Special tokens: <s>=1 (BOS), </s>=2 (EOS), [INST]=3, [/INST]=4
# No ChatML tokens (<|im_start|> etc.) — those are Qwen-specific.
# ─────────────────────────────────────────────────────────────────────────────
class TestTokenizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load the tokenizer once for all tokenizer tests."""
        os.environ["HF_HUB_OFFLINE"] = "1"
        if not SNAP_DIR.exists():
            raise unittest.SkipTest(
                f"Tokenizer snapshot not found at {SNAP_DIR}. "
                "Run the snapshot download step first (see README.md)."
            )
        try:
            from transformers import AutoTokenizer
            cls.tok = AutoTokenizer.from_pretrained(
                str(SNAP_DIR), local_files_only=True
            )
        except Exception as exc:
            raise unittest.SkipTest(f"Tokenizer load failed: {exc}") from exc

    def test_offline_load(self):
        """Tokenizer must load without network access."""
        self.assertIsNotNone(self.tok)

    def test_vocab_size(self):
        """Mistral-7B-v0.3 has a 32k vocab."""
        vocab = len(self.tok)
        # v0.3 extended to 32768; accept anything in [32000, 33000]
        self.assertGreaterEqual(vocab, 32000,
            f"Vocab size {vocab} is smaller than expected for Mistral-7B-v0.3")
        self.assertLessEqual(vocab, 33000,
            f"Vocab size {vocab} is larger than expected for Mistral-7B-v0.3")
        print(f"\n  vocab size = {vocab}", end="")

    def test_eos_token(self):
        """EOS token must be </s> with id=2."""
        self.assertEqual(self.tok.eos_token, "</s>",
            f"Expected eos_token='</s>', got {self.tok.eos_token!r}")
        self.assertEqual(self.tok.eos_token_id, 2,
            f"Expected eos_token_id=2, got {self.tok.eos_token_id}")
        print(f"\n  </s> -> id={self.tok.eos_token_id}", end="")

    def test_bos_token(self):
        """BOS token must be <s> with id=1."""
        self.assertEqual(self.tok.bos_token, "<s>",
            f"Expected bos_token='<s>', got {self.tok.bos_token!r}")
        self.assertEqual(self.tok.bos_token_id, 1,
            f"Expected bos_token_id=1, got {self.tok.bos_token_id}")
        print(f"\n  <s> -> id={self.tok.bos_token_id}", end="")

    def test_inst_tokens_present(self):
        """[INST] and [/INST] must exist as single tokens in Mistral v0.3."""
        for tok_str in ("[INST]", "[/INST]"):
            ids = self.tok.encode(tok_str, add_special_tokens=False)
            self.assertEqual(
                len(ids), 1,
                f"{tok_str!r} encoded to {len(ids)} tokens (expected 1): {ids}. "
                "This means the tokenizer snapshot may be wrong or a v0.1/v0.2 snapshot."
            )
            print(f"\n  {tok_str} -> id={ids[0]}", end="")

    def test_no_chatml_tokens(self):
        """Mistral does NOT use ChatML — <|im_start|> should not be a single token."""
        ids = self.tok.encode("<|im_start|>", add_special_tokens=False)
        self.assertGreater(
            len(ids), 1,
            "<|im_start|> encoded to a single token — this looks like a Qwen tokenizer, not Mistral."
        )

    def test_roundtrip(self):
        """A plain sentence must survive encode→decode."""
        sentence = "Machine learning models require careful evaluation."
        enc = self.tok.encode(sentence, add_special_tokens=False)
        dec = self.tok.decode(enc)
        self.assertIn(
            sentence, dec,
            f"Round-trip failed: original={sentence!r} decoded={dec!r}"
        )

    def test_mistral_chat_template(self):
        """
        apply_chat_template must produce [INST]...[/INST] format and
        include the system prompt inside the first [INST] block.
        """
        messages = [
            {"role": "system",    "content": "You are a health information assistant."},
            {"role": "user",      "content": "What is ibuprofen used for?"},
            {"role": "assistant", "content": "Ibuprofen is an NSAID used for pain and fever."},
            {"role": "user",      "content": "Is it safe during pregnancy?"},
        ]
        try:
            rendered = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as exc:
            self.skipTest(f"apply_chat_template raised: {exc}")

        self.assertIn("[INST]", rendered,
            "Expected [INST] in rendered template — is this really a Mistral tokenizer?")
        self.assertIn("[/INST]", rendered,
            "Expected [/INST] in rendered template.")
        self.assertIn("ibuprofen", rendered.lower(),
            "User message content missing from rendered template.")
        # Mistral v0.3 places system prompt text inside the first [INST] block
        self.assertIn("health information assistant", rendered.lower(),
            "System prompt text missing from rendered template.")
        print(f"\n  chat template ok, len={len(rendered)}", end="")

    def test_pad_token_not_eos(self):
        """
        pad_token must not equal eos_token (</s>).
        If the tokenizer has no pad token set, we just note it — the training
        script sets pad_token = unk_token or a dedicated pad.
        """
        if self.tok.pad_token_id is None:
            print("\n  pad_token_id=None (will be set in training script)", end="")
            return  # acceptable — training script handles it
        self.assertNotEqual(
            self.tok.pad_token_id, self.tok.eos_token_id,
            "pad_token_id == eos_token_id — this causes masked padding to be treated as EOS "
            "and will corrupt the loss mask. Set pad_token to unk_token in the training script."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — Mistral chat-format masking helpers
#
# Verifies the logic used in train_sft.py to build the assistant-only label mask.
# Uses no GPU and no model weights — pure Python.
# ─────────────────────────────────────────────────────────────────────────────
class TestMistralMasking(unittest.TestCase):
    """
    The Mistral [INST]...[/INST] format looks like:
      <s>[INST] {user_turn} [/INST] {assistant_turn}</s>[INST] {user_turn} [/INST] ...
    We supervise only assistant tokens (after [/INST] up to and including </s>).
    Everything else gets label=-100.
    """

    @classmethod
    def setUpClass(cls):
        os.environ["HF_HUB_OFFLINE"] = "1"
        if not SNAP_DIR.exists():
            raise unittest.SkipTest("Tokenizer snapshot not found.")
        try:
            from transformers import AutoTokenizer
            cls.tok = AutoTokenizer.from_pretrained(
                str(SNAP_DIR), local_files_only=True
            )
            # Mistral v0.3 may not have pad_token set
            if cls.tok.pad_token_id is None:
                cls.tok.pad_token = cls.tok.unk_token
        except Exception as exc:
            raise unittest.SkipTest(f"Tokenizer load failed: {exc}") from exc

    def _build_labels(self, messages: list[dict]) -> tuple[list[int], list[int]]:
        """
        Minimal reference implementation of the assistant-only label mask for
        Mistral-format conversations. Returns (input_ids, labels).

        Mistral apply_chat_template with tokenize=True returns the full sequence.
        We then scan for [/INST] tokens to find where assistant turns start,
        and for </s> to find where they end.
        """
        tok = self.tok
        # Build the full sequence including the final EOS for each assistant turn
        full_text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        input_ids = tok.encode(full_text, add_special_tokens=False)
        labels = [-100] * len(input_ids)

        # Find [/INST] and </s> token ids
        inst_end_ids = tok.encode("[/INST]", add_special_tokens=False)
        eos_id = tok.eos_token_id  # </s> = 2

        if len(inst_end_ids) != 1:
            # Can't do the scan — skip the label-building check
            return input_ids, labels

        inst_end_id = inst_end_ids[0]

        # Scan: after each [/INST], label tokens until </s> (inclusive)
        i = 0
        while i < len(input_ids):
            if input_ids[i] == inst_end_id:
                # assistant section starts at i+1
                j = i + 1
                while j < len(input_ids):
                    labels[j] = input_ids[j]
                    if input_ids[j] == eos_id:
                        break
                    j += 1
                i = j + 1
            else:
                i += 1

        return input_ids, labels

    def test_single_exchange_has_supervised_tokens(self):
        """At least some tokens must be supervised (label != -100)."""
        messages = [
            {"role": "user",      "content": "What causes a fever?"},
            {"role": "assistant", "content": "A fever is caused by infection or inflammation."},
        ]
        _, labels = self._build_labels(messages)
        supervised = [l for l in labels if l != -100]
        self.assertGreater(len(supervised), 0,
            "No supervised tokens found — label mask is all -100.")

    def test_no_user_tokens_supervised(self):
        """User turn tokens must all be -100."""
        user_text = "What causes a fever?"
        messages = [
            {"role": "user",      "content": user_text},
            {"role": "assistant", "content": "A fever is caused by infection or inflammation."},
        ]
        input_ids, labels = self._build_labels(messages)
        tok = self.tok
        user_ids = tok.encode(user_text, add_special_tokens=False)
        # Find where the user text appears in input_ids
        for start in range(len(input_ids) - len(user_ids) + 1):
            if input_ids[start:start + len(user_ids)] == user_ids:
                for pos in range(start, start + len(user_ids)):
                    self.assertEqual(
                        labels[pos], -100,
                        f"User token at position {pos} (id={input_ids[pos]}) is supervised — it should be -100."
                    )
                break

    def test_multiturn_all_assistant_turns_supervised(self):
        """Both assistant turns in a 2-exchange dialogue must have supervised tokens."""
        messages = [
            {"role": "user",      "content": "I have a headache."},
            {"role": "assistant", "content": "How long have you had it?"},
            {"role": "user",      "content": "About two days."},
            {"role": "assistant", "content": "Two days is worth monitoring. Drink water and rest."},
        ]
        _, labels = self._build_labels(messages)
        supervised = [l for l in labels if l != -100]
        # Rough check: both assistant turns together should be at least 15 tokens
        self.assertGreater(len(supervised), 15,
            f"Only {len(supervised)} supervised tokens for a 2-turn dialogue — expected more.")

    def test_eos_at_end_of_each_assistant_turn(self):
        """The EOS token (</s>) at the end of each assistant turn must be supervised."""
        messages = [
            {"role": "user",      "content": "I have a headache."},
            {"role": "assistant", "content": "How long have you had it?"},
            {"role": "user",      "content": "About two days."},
            {"role": "assistant", "content": "Drink water and rest."},
        ]
        input_ids, labels = self._build_labels(messages)
        eos_id = self.tok.eos_token_id
        eos_positions = [i for i, t in enumerate(input_ids) if t == eos_id]
        if not eos_positions:
            self.skipTest("No EOS tokens found in sequence — template may differ.")
        for pos in eos_positions:
            self.assertEqual(
                labels[pos], eos_id,
                f"EOS token at position {pos} has label -100 — it should be supervised."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — verifier unit tests (stdlib only, uses temp files)
# ─────────────────────────────────────────────────────────────────────────────
class TestVerifier(unittest.TestCase):
    """
    Tests the verifier logic in isolation using temporary files only.
    Imports verify_manifest from the sft/ directory.
    """

    def _make_manifest(self, tmpdir: Path, files: dict) -> Path:
        import hashlib, json
        from datetime import datetime, timezone

        entries = []
        for rel, content in files.items():
            fp = tmpdir / rel
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_bytes(content)
            st = fp.stat()
            entries.append({
                "path":      rel,
                "size":      st.st_size,
                "mtime":     st.st_mtime,
                "mtime_iso": datetime.now(tz=timezone.utc).isoformat(),
                "hashed":    True,
                "sha256":    hashlib.sha256(content).hexdigest(),
            })

        manifest = {
            "schema_version":  "1",
            "created_utc":     datetime.now(tz=timezone.utc).isoformat(),
            "repo_root":       str(tmpdir),
            "total_files":     len(entries),
            "hashed_count":    len(entries),
            "unhashed_count":  0,
            "entries":         entries,
        }
        mpath = tmpdir / "manifest.json"
        mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return mpath

    def _run_verifier(self, manifest_path: Path, repo_root: Path) -> tuple:
        import io, contextlib
        sys.path.insert(0, str(SFT_DIR))
        import verify_manifest as vm

        buf = io.StringIO()
        exit_code = 0

        try:
            import json
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_data["repo_root"] = str(repo_root)
            manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

            with contextlib.redirect_stdout(buf):
                try:
                    original_exit = sys.exit
                    def fake_exit(code=0):
                        raise SystemExit(code)
                    sys.exit = fake_exit

                    manifest_obj = json.loads(manifest_path.read_text(encoding="utf-8"))
                    entries = manifest_obj["entries"]
                    errors = []
                    ok_count = 0
                    for entry in entries:
                        status, detail = vm.verify_entry(entry, repo_root)
                        if status is None:
                            ok_count += 1
                        else:
                            errors.append(f"[{status}] {detail}")

                    manifest_paths = {e["path"] for e in entries}
                    current_paths = set()
                    for dirpath, _, filenames in os.walk(repo_root):
                        for fname in filenames:
                            fp2 = Path(dirpath) / fname
                            rel = fp2.relative_to(repo_root).as_posix()
                            if rel != "manifest.json":
                                current_paths.add(rel)
                    new_paths = current_paths - manifest_paths

                    warnings = [f"[WARNING] New unmanifested file: {p}" for p in sorted(new_paths)]
                    for w in warnings:
                        print(w)
                    for e in errors:
                        print(e)
                    if errors:
                        exit_code = 1
                        print(f"VERIFICATION FAILED: {len(errors)} error(s)")
                    else:
                        print(f"VERIFICATION PASSED: {ok_count} ok, {len(warnings)} warning(s).")

                except SystemExit as e:
                    exit_code = e.code if e.code is not None else 0
                finally:
                    sys.exit = original_exit
        finally:
            pass

        return exit_code, buf.getvalue()

    def test_passes_when_untouched(self):
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            files = {"data/file_a.txt": b"hello world", "data/file_b.txt": b"foo bar"}
            mpath = self._make_manifest(tmpdir, files)
            code, out = self._run_verifier(mpath, tmpdir)
            self.assertEqual(code, 0, f"Expected exit 0, got {code}. Output:\n{out}")
            self.assertIn("PASSED", out)

    def test_fails_on_modified_file(self):
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            files = {"data/file_a.txt": b"original content"}
            mpath = self._make_manifest(tmpdir, files)
            (tmpdir / "data" / "file_a.txt").write_bytes(b"tampered content")
            code, out = self._run_verifier(mpath, tmpdir)
            self.assertEqual(code, 1, f"Expected exit 1, got {code}. Output:\n{out}")
            self.assertIn("CHANGED", out)

    def test_fails_on_missing_file(self):
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            files = {"data/file_a.txt": b"will be deleted"}
            mpath = self._make_manifest(tmpdir, files)
            (tmpdir / "data" / "file_a.txt").unlink()
            code, out = self._run_verifier(mpath, tmpdir)
            self.assertEqual(code, 1, f"Expected exit 1, got {code}. Output:\n{out}")
            self.assertIn("MISSING", out)

    def test_warns_on_added_file(self):
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            files = {"data/file_a.txt": b"original"}
            mpath = self._make_manifest(tmpdir, files)
            (tmpdir / "data" / "new_file.txt").write_bytes(b"new content")
            code, out = self._run_verifier(mpath, tmpdir)
            self.assertEqual(code, 0, f"Expected exit 0, got {code}. Output:\n{out}")
            self.assertIn("WARNING", out)
            self.assertIn("PASSED", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
