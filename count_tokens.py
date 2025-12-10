"""
Count tokenizer tokens for Meditron JSONL datasets.

Datasets:
- miriad
- mediset (meadow + medqa + pubmedqa + medmcqa + wikidoc_qa)
- medtext
- guidelines
- pubmed

Usage:
    python count_tokens.py
"""

import json
from pathlib import Path

from transformers import AutoTokenizer


BASE_MODEL = "/capstor/store/cscs/swissai/a127/apertus/huggingface/Apertus8B"
STORAGE_ROOT = Path("/capstor/store/cscs/swissai/a127")


DATASETS = {
    "miriad": STORAGE_ROOT / "meditron/datasets/masked/miriad/miriad-4.4M.jsonl",
    "mediset_meadow": STORAGE_ROOT / "meditron/datasets/masked/meadow/meadow.jsonl",
    "mediset_medqa": STORAGE_ROOT / "meditron/datasets/masked/medqa/medqa.jsonl",
    "mediset_pubmedqa": STORAGE_ROOT / "meditron/datasets/masked/pubmedqa/pubmedqa.jsonl",
    "mediset_medmcqa": STORAGE_ROOT / "meditron/datasets/masked/medmcqa/medmcqa.jsonl",
    "mediset_wikidoc_qa": STORAGE_ROOT / "meditron/datasets/masked/wikidoc_qa/wikidoc_qa.jsonl",
    "medtext": STORAGE_ROOT / "meditron/datasets/masked/medtext/medtext.jsonl",
    "guidelines": STORAGE_ROOT / "meditron/datasets/pretrain/guidelines/guidelines.jsonl",
    "pubmed": STORAGE_ROOT / "meditron/datasets/pretrain/pubmed/pubmed_3B.jsonl",
}


def load_tokenizer():
    return AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)


def iter_chat_text(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            conv = obj.get("conversations") or []
            texts = []
            for m in conv:
                v = m.get("value")
                if isinstance(v, str):
                    texts.append(v)
            if texts:
                yield "\n".join(texts)


def iter_completion_text(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            txt = obj.get("text")
            if isinstance(txt, str):
                yield txt


def count_tokens(tokenizer, path: Path, mode: str):
    total = 0
    if mode == "chat":
        iterator = iter_chat_text(path)
    else:
        iterator = iter_completion_text(path)
    for text in iterator:
        total += len(tokenizer.encode(text, add_special_tokens=False))
    return total


def main():
    tokenizer = load_tokenizer()

    chat_files = [
        "miriad",
        "mediset_meadow",
        "mediset_medqa",
        "mediset_pubmedqa",
        "mediset_medmcqa",
        "mediset_wikidoc_qa",
        "medtext",
    ]
    completion_files = ["guidelines", "pubmed"]

    tokens = {}

    for name in chat_files:
        path = DATASETS[name]
        t = count_tokens(tokenizer, path, mode="chat")
        tokens[name] = t
        print(f"{name}: {t:,} tokens ({t/1e9:.3f}B)")

    mediset_total = sum(tokens[name] for name in chat_files if name.startswith("mediset_"))
    print(f"\nmediset (combined): {mediset_total:,} tokens ({mediset_total/1e9:.3f}B)")

    for name in completion_files:
        path = DATASETS[name]
        t = count_tokens(tokenizer, path, mode="completion")
        tokens[name] = t
        print(f"{name}: {t:,} tokens ({t/1e9:.3f}B)")


if __name__ == "__main__":
    main()
