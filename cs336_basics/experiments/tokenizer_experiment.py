import pickle
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cs336_basics.bpe_tokenizer import bpe_tokenizer

DATA_ROOT = PROJECT_ROOT / "data"
OUT = PROJECT_ROOT / "cs336_basics" / "experiments" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

random.seed(0)


def load_tokenizer(name):
    with open(OUT / f"{name}.pkl", "rb") as f:
        vocab, merge = pickle.load(f)
    return bpe_tokenizer(vocab, merge, special_tokens=["<|endoftext|>"])


def sample_docs(path, n=10):
    text = path.read_text(encoding="utf-8")
    docs = [doc for doc in text.split("<|endoftext|>") if doc.strip()]
    return random.sample(docs, n)


def train_and_save(input_path, vocab_size, name):
    tokenizer = bpe_tokenizer()
    vocab, merge = tokenizer.train_bpe(input_path, vocab_size, special_tokens=["<|endoftext|>"])
    with open(OUT / f"{name}.pkl", "wb") as f:
        pickle.dump((vocab, merge), f)


def compression_rate(tokenizer, docs):
    total_bytes = sum(len(doc.encode("utf-8")) for doc in docs)
    total_tokens = sum(len(tokenizer.encode(doc)) for doc in docs)
    if total_tokens == 0:
        raise ValueError("No tokens found")
    return total_bytes / total_tokens


if __name__ == "__main__":
    # train_and_save(DATA_ROOT / "TinyStoriesV2-GPT4-train.txt", 10_000, "tinystories")
    # train_and_save(DATA_ROOT / "owt_train.txt", 32_000, "owt")

    tiny_tok = load_tokenizer("tinystories")
    owt_tok = load_tokenizer("owt")

    tiny_docs = sample_docs(DATA_ROOT / "TinyStoriesV2-GPT4-valid.txt")
    owt_docs = sample_docs(DATA_ROOT / "owt_valid.txt")

    print("TinyStories docs with TinyStories tokenizer:", compression_rate(tiny_tok, tiny_docs))
    print("OWT docs with OWT tokenizer:", compression_rate(owt_tok, owt_docs))
    print("OWT docs with TinyStories tokenizer:", compression_rate(tiny_tok, owt_docs))
