# Tokenizer Experiments

## Compression Ratios

Measured on 10 sampled documents from each validation set:

```text
TinyStories docs with TinyStories tokenizer: 4.101558301988178 bytes/token
OWT docs with OWT tokenizer: 4.321614479058221 bytes/token
OWT docs with TinyStories tokenizer: 3.0794339191965303 bytes/token
```

Answer (a): On 10 sampled documents, the TinyStories tokenizer achieves a compression ratio of 4.10 bytes/token on TinyStories, while the OpenWebText tokenizer achieves 4.32 bytes/token on OpenWebText.

Answer (b): When tokenizing the OpenWebText sample with the TinyStories tokenizer, the compression ratio drops to 3.08 bytes/token, meaning it uses more tokens per byte. Qualitatively, the TinyStories tokenizer generalizes worse to the broader OpenWebText domain.

## Throughput Benchmark

Benchmark script:

```python
import pickle
import time
from pathlib import Path

from cs336_basics.bpe_tokenizer import bpe_tokenizer

ROOT = Path("/mnt/data3/muxing.fyy/cs336/assignment1-basics")
OUT = ROOT / "cs336_basics/experiments/outputs"
DATA = ROOT / "data"

with open(OUT / "owt.pkl", "rb") as f:
    vocab, merges = pickle.load(f)

tok = bpe_tokenizer(vocab, merges, ["<|endoftext|>"])

path = DATA / "owt_valid.txt"
raw = path.read_bytes()[:5_000_000]
text = raw.decode("utf-8", errors="ignore")
byte_count = len(text.encode("utf-8"))

_ = tok.encode(text[:10_000])
start = time.perf_counter()
ids = tok.encode(text)
elapsed = time.perf_counter() - start

throughput = byte_count / elapsed
pile_bytes = 825 * 1_000_000_000
seconds = pile_bytes / throughput

print(f"bytes={byte_count}")
print(f"tokens={len(ids)}")
print(f"elapsed_sec={elapsed:.6f}")
print(f"throughput_bytes_per_sec={throughput:.2f}")
print(f"pile_seconds={seconds:.2f}")
print(f"pile_hours={seconds / 3600:.2f}")
print(f"pile_days={seconds / 86400:.2f}")
```

Output:

```text
bytes=5000000
tokens=1128966
elapsed_sec=8.823942
throughput_bytes_per_sec=566640.14
pile_seconds=1455950.50
pile_hours=404.43
pile_days=16.85
```

Answer (c): Benchmarking the OpenWebText tokenizer on a 5MB OWT validation slice gave about 5.67e5 bytes/sec, or 0.57 MB/s. At that rate, tokenizing 825GB of text would take about 1.46e6 seconds, roughly 16.9 days on one process.

Answer (d): `uint16` is appropriate because both vocabularies are smaller than `2^16 = 65,536`: TinyStories uses 10K tokens and OpenWebText uses 32K tokens. It stores each token ID in 2 bytes, saving space compared with `uint32` or `int64` while still covering every possible token ID.
