#!/usr/bin/env python3
"""Run the trainer's pre-training checks on a corpus, without training. Tokenizer only.

The same checks train_student.py runs before its first step (sl_da/train.py::load_corpus):

  NO SYSTEM PROMPT  on every record: no chat control tokens, rendered prefix is the bare
                    user-turn header (no system turn, not even Qwen's default), and no
                    sentence of any known system prompt (initial_checks/configs/*.txt, the
                    corpus's .meta.json prompt, Qwen's default) in the text.
  LOSS MASK         on every example: prompt tokens all -100, response tokens all
                    supervised, each span decodes back to exactly the prompt / response.

Exit 0 if every record passes, non-zero otherwise. Runs on a laptop: Qwen2.5 shares one
tokenizer and chat template across sizes, so the 0.5B tokenizer checks a 14B corpus.

    python scripts/check_training_data.py --base unsloth/Qwen2.5-0.5B-Instruct \
      --corpus ../data/corpora/corpus_owl.jsonl
"""
import argparse, collections, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl_da.train import TrainConfig, load_corpus

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True, help="model whose tokenizer the student will use")
ap.add_argument("--corpus", required=True)
ap.add_argument("--max-len", type=int, default=1024, help="must match train_student.py")
a = ap.parse_args()

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.base)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
ex, ids, manifest, checks = load_corpus(
    a.corpus, tok, TrainConfig(base=a.base, corpus=a.corpus, out_dir="", max_len=a.max_len))
drops = collections.Counter(m.get("drop_reason") for m in manifest if not m["used"])
print(f"\n  PASS  {len(ex):,}/{len(manifest):,} records trainable"
      + (f"; dropped: {dict(drops)}" if drops else ""))
