"""
GPT-2 (124M) Training Script — Google Colab Background Edition
==============================================================

Usage (in a Colab cell):
    1. Upload this file + helloswag.py + data/ folder to:
         My Drive  ▸  Colab Notebooks/

    2. Mount Drive and run in background:
         !nohup python "/content/drive/MyDrive/Colab Notebooks/train_colab.py" > "/content/drive/MyDrive/Colab Notebooks/log/train_output.log" 2>&1 &

    3. Monitor progress:
         !tail -f "/content/drive/MyDrive/Colab Notebooks/log/train_output.log"

All checkpoints and logs are saved to:
    /content/drive/MyDrive/Colab Notebooks/log/
"""

# ── Imports ──────────────────────────────────────────────────────────────────
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import tiktoken
import inspect
import numpy as np
import os
import sys
import time

# ── Configuration ────────────────────────────────────────────────────────────
# Base directory: everything lives under Colab Notebooks on Google Drive
BASE_DIR = "/content/drive/MyDrive/Colab Notebooks"
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR  = os.path.join(BASE_DIR, "log")

# Add BASE_DIR to path so we can import helloswag
sys.path.insert(0, BASE_DIR)

# ── Verify Google Drive is mounted ───────────────────────────────────────────
# NOTE: Mount Drive in your Colab cell BEFORE running this script.
#       drive.mount() does not work in standalone Python scripts.
if os.path.isdir("/content/drive/MyDrive"):
    print("✅ Google Drive is mounted.")
else:
    print("❌ Google Drive is NOT mounted! Run this in a Colab cell first:")
    print('   from google.colab import drive; drive.mount("/content/drive")')
    sys.exit(1)

# Make sure the log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# ── Import HellaSwag evaluation helpers ──────────────────────────────────────
# NOTE: helloswag.py uses os.path.dirname(__file__) to find hellaswag/ data.
#       Make sure you also upload the hellaswag/ folder to Colab Notebooks/.
from helloswag import render_example, iterate_examples

# ── Device setup ─────────────────────────────────────────────────────────────
assert torch.cuda.is_available(), "CUDA is required — use a Colab GPU runtime!"
device = "cuda"
print(f"🖥️  Device       : {torch.cuda.get_device_name()}")
print(f"🧠  VRAM         : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ═══════════════════════════════════════════════════════════════════════════════
#                              MODEL DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # Flash Attention — never materializes the N×N attention matrix in HBM
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu   = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn  = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp   = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer:    int = 12
    n_head:     int = 12
    n_embd:     int = 768


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if hasattr(module, "NANOGPT_SCALE_INIT"):
            std *= (2 * self.config.n_layer) ** -0.5
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        )
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params   = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params,   "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay   = sum(p.numel() for p in decay_params)
        num_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"  decayed param tensors  : {len(decay_params):,}, with {num_decay:,} parameters")
        print(f"  non-decayed param tensors: {len(nodecay_params):,}, with {num_nodecay:,} parameters")
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print(f"  using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused,
        )
        return optimizer


# ═══════════════════════════════════════════════════════════════════════════════
#                              HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_most_likely_row(tokens, mask, logits):
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction="none")
    shift_losses = shift_losses.view(tokens.size(0), -1)
    shift_mask = (mask[..., 1:]).contiguous()
    masked_shift_losses = shift_losses * shift_mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    pred_norm = avg_loss.argmin().item()
    return pred_norm


def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    """Simple data loader that reads pre-tokenized .npy shards from disk."""

    def __init__(self, B, T, split):
        self.B = B
        self.T = T
        assert split in {"train", "val"}
        shards = sorted([
            os.path.join(DATA_DIR, s)
            for s in os.listdir(DATA_DIR)
            if split in s and s.endswith(".npy")
        ])
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split '{split}' in {DATA_DIR}"
        print(f"  found {len(shards)} shards for split '{split}'")
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        buf = buf.to(device=device)
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = 0
        return x, y


# ═══════════════════════════════════════════════════════════════════════════════
#                           LEARNING RATE SCHEDULE
# ═══════════════════════════════════════════════════════════════════════════════

max_lr       = 6e-4
min_lr       = max_lr * 0.1
warmup_steps = 95
max_steps    = 950


def get_lr(it):
    """Cosine decay with linear warmup."""
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# ═══════════════════════════════════════════════════════════════════════════════
#                           CHECKPOINT UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

CHECKPOINT_PATH = os.path.join(LOG_DIR, "checkpoint.pt")


def save_checkpoint(model, optimizer, step, train_loss, val_loss):
    """Save model, optimizer state, and current step to Drive."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"  💾 Checkpoint saved at step {step}")


def load_checkpoint(model, optimizer):
    """Resume from the latest checkpoint if it exists."""
    if os.path.exists(CHECKPOINT_PATH):
        print(f"  📂 Resuming from checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint["step"] + 1
        print(f"  ↳ Resuming from step {start_step}")
        return start_step
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
#                              TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  GPT-2 Training — Colab Background Edition")
    print("=" * 70)

    torch.manual_seed(1337)
    torch.set_float32_matmul_precision("high")

    # ── Batch sizing ─────────────────────────────────────────────────────────
    total_batch_size = 524288  # 2**19, ~0.5M tokens
    B = 8   # micro batch size  (adjust if OOM on your Colab GPU)
    T = 1024  # sequence length
    grad_accum_steps = total_batch_size // (B * T)
    print(f"\n📦 Batch config:")
    print(f"  total batch size              : {total_batch_size:,} tokens")
    print(f"  micro batch (B×T)             : {B}×{T} = {B*T:,}")
    print(f"  gradient accumulation steps   : {grad_accum_steps}")

    # ── Data loaders ─────────────────────────────────────────────────────────
    print(f"\n📂 Data directory: {DATA_DIR}")
    train_loader = DataLoaderLite(B=B, T=T, split="train")
    val_loader   = DataLoaderLite(B=B, T=T, split="val")

    # ── Model ────────────────────────────────────────────────────────────────
    # vocab_size 50304 is a "nice" power-of-two-friendly number for CUDA kernels
    print("\n🧠 Initializing model...")
    model = GPT(GPTConfig(vocab_size=50304))
    model.to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optional: torch.compile for extra speed (requires Triton / Linux)
    use_compile = False
    if use_compile:
        model = torch.compile(model)

    # ── Optimizer ────────────────────────────────────────────────────────────
    print("\n⚙️  Configuring optimizer...")
    optimizer = model.configure_optimizers(
        weight_decay=0.1, learning_rate=6e-4, device_type=device,
    )

    # ── Resume from checkpoint if available ──────────────────────────────────
    start_step = load_checkpoint(model, optimizer)

    # ── Logging ──────────────────────────────────────────────────────────────
    log_file = os.path.join(LOG_DIR, "log.txt")
    enc = tiktoken.get_encoding("gpt2")

    print(f"\n🚀 Starting training from step {start_step} to {max_steps}...\n")
    val_loss_accum = None  # track for checkpoint saving
    sys.stdout.flush()

    for i in range(start_step, max_steps):
        t0 = time.time()
        last_step = (i == max_steps - 1)

        # ── Validation loss (every 10 steps) ─────────────────────────────────
        if i % 10 == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    with torch.autocast(device_type=device, dtype=torch.float16):
                        logits, loss = model(x, y)
                    val_loss_accum += loss.item() / val_loss_steps
            print(f"  📊 Validation loss: {val_loss_accum:.4f}")

        # ── HellaSwag evaluation (every 250 steps) ───────────────────────────
        if (i % 250 == 0 or last_step) and (not use_compile):
            model.eval()
            num_correct_norm = 0
            num_total = 0
            for j, example in enumerate(iterate_examples("val")):
                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(device)
                mask   = mask.to(device)
                with torch.no_grad():
                    with torch.autocast(device_type=device, dtype=torch.float16):
                        logits, loss = model(tokens)
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)
            acc_norm = num_correct_norm / num_total
            print(f"  🎯 HellaSwag accuracy: {num_correct_norm}/{num_total} = {acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"step {i} hella {acc_norm:.4f}\n")

        # ── Sample generation (every 10 steps after step 0) ──────────────────
        if i > 0 and i % 10 == 0 and (not use_compile):
            model.eval()
            num_return_sequences = 4
            max_length = 32
            tokens = enc.encode("Hello , I am a language model")
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
            xgen = tokens.to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42)
            while xgen.size(1) < max_length:
                with torch.no_grad():
                    logits, loss = model(xgen)
                    logits = logits[:, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                    xcol = torch.gather(topk_indices, -1, ix)
                    xgen = torch.cat((xgen, xcol), dim=1)
            for seq_i in range(num_return_sequences):
                decoded = enc.decode(xgen[seq_i, :max_length].tolist())
                print(f"  💬 sample {seq_i}: {decoded}")

        # ── Training step ────────────────────────────────────────────────────
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            with torch.autocast(device_type=device, dtype=torch.float16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach().item()
            loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(i)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()
        torch.cuda.synchronize()

        t1 = time.time()
        dt = (t1 - t0) * 1000  # ms
        tokens_processed = B * T * grad_accum_steps
        tokens_per_sec = tokens_processed / (t1 - t0)

        print(
            f"step {i:>4d} | loss {loss_accum:.4f} | norm {norm:.4f} | "
            f"lr {lr:.2e} | {dt:.0f}ms | {tokens_per_sec:.0f} tok/s"
        )
        with open(log_file, "a") as f:
            f.write(f"{i} train {loss_accum:.6f}\n")

        # ── Checkpoint every 50 steps + on last step ─────────────────────────
        if (i % 50 == 0 and i > 0) or last_step:
            save_checkpoint(model, optimizer, i, loss_accum, val_loss_accum if i % 10 == 0 else None)

        sys.stdout.flush()  # ensure nohup output is written immediately

    # ── Save final model ─────────────────────────────────────────────────────
    final_model_path = os.path.join(LOG_DIR, "gpt2_final.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"\n✅ Training complete! Final model saved to: {final_model_path}")


if __name__ == "__main__":
    main()
