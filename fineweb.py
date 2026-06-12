"""
FineWeb-Edu dataset (for SRS pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

Downloads and tokenizes the data, then saves token shards to disk.
Run:
    python fineweb.py
"""

import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# ------------------------------------------
local_dir = "data"
remote_name = "sample-10BT"

shard_size = int(5e7)   # 50M tokens per shard
max_shards = 10         # stop after writing 10 shards

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# Streaming mode prevents Hugging Face from downloading the whole dataset up front
fw = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name=remote_name,
    split="train",
    streaming=True,
)

# tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]

def tokenize(doc):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens, dtype=np.int64)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    return tokens_np.astype(np.uint16)

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

shard_index = 0
token_count = 0
all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
progress_bar = None

for doc in fw:
    tokens = tokenize(doc)
    pos = 0

    while pos < len(tokens):
        if shard_index >= max_shards:
            break

        remaining = shard_size - token_count
        take = min(remaining, len(tokens) - pos)

        all_tokens_np[token_count:token_count + take] = tokens[pos:pos + take]
        token_count += take
        pos += take

        if progress_bar is None:
            progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
        progress_bar.update(take)

        # shard is full, write it out
        if token_count == shard_size:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"data_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np)

            if progress_bar is not None:
                progress_bar.close()
                progress_bar = None

            shard_index += 1
            token_count = 0

            if shard_index >= max_shards:
                break

            all_tokens_np = np.empty((shard_size,), dtype=np.uint16)

    if shard_index >= max_shards:
        break

# write the final partial shard, if any
if shard_index < max_shards and token_count > 0:
    split = "val" if shard_index == 0 else "train"
    filename = os.path.join(DATA_CACHE_DIR, f"data_{split}_{shard_index:06d}")
    write_datafile(filename, all_tokens_np[:token_count])

if progress_bar is not None:
    progress_bar.close()