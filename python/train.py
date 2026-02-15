"""
Training loop for GrassLM on Wikitext-2.

Command:
    python train.py --model_name GrassLM-10M [--epochs 30] [--batch_size 32]
                    [--lr 3e-4] [--seq_len 128] [--d_model 256] [--n_layers 6]
                    [--d_reduce 32] [--d_ff 1024] [--dropout 0.1]
                    [--warmup_steps 2000] [--weight_decay 0.01] [--seed 42]

    # Or with explicit checkpoint dir:
    python train.py --checkpoint_dir path/to/checkpoints [...]
"""

import argparse
import math
import os
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from grasslm.data import create_dataloaders
from grasslm.model import GrassLM
from grasslm.registry import models_dir, save_config


def get_device() -> torch.device:
    # Select best available device: MPS (Apple Silicon) > CUDA > CPU
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_lr_schedule(
    optimizer: AdamW,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    # Linear warmup for `warmup_steps`, then cosine decay to zero.

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model: nn.Module, val_loader, device: torch.device) -> float:
    # Compute perplexity on a validation/test set.
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids)  # (B, L, V)
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        total_loss += loss.item()
        total_tokens += labels.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity


def train(args: argparse.Namespace) -> None:
    # Main training loop
    torch.manual_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    # Data
    print("Loading Wikitext-2 data...")
    train_loader, val_loader, _ = create_dataloaders(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    model = GrassLM(
        vocab_size=30522,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_reduce=args.d_reduce,
        d_ff=args.d_ff,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
    )

    # LR schedule
    total_steps = len(train_loader) * args.epochs
    scheduler = create_lr_schedule(optimizer, args.warmup_steps, total_steps)

    # Loss
    loss_fn = nn.CrossEntropyLoss()

    # Checkpointing
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_ppl = float("inf")
    start_epoch = 1
    global_step = 0

    # Resume from checkpoint if provided
    if args._resume_ckpt is not None:
        ckpt = args._resume_ckpt
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)
        best_val_ppl = ckpt.get("val_ppl", float("inf"))
        print(f"Resumed from epoch {ckpt['epoch']}, val PPL {best_val_ppl:.2f}")

    # Training
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids)  # (B, L, V)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            batch_tokens = labels.numel()
            epoch_loss += loss.item() * batch_tokens
            epoch_tokens += batch_tokens
            global_step += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        epoch_time = time.time() - t0
        avg_train_loss = epoch_loss / epoch_tokens

        # Validation
        val_ppl = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train loss: {avg_train_loss:.4f} | "
            f"Val PPL: {val_ppl:.2f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save best checkpoint
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_ppl": val_ppl,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"  -> New best model saved (val PPL: {val_ppl:.2f})")

        # Save latest checkpoint every epoch
        ckpt_path = os.path.join(args.checkpoint_dir, "latest.pt")
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_ppl": val_ppl,
                "args": vars(args),
            },
            ckpt_path,
        )

    print(f"\nTraining complete. Best validation perplexity: {best_val_ppl:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GrassLM on Wikitext-2")

    # Model
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--d_reduce", type=int, default=32)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Data
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument(
        "--model_name", type=str, default=None,
        help="Model name (e.g. GrassLM-10M). Saves to models/<name>/checkpoints/",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default=None,
        help="Explicit checkpoint directory (overrides --model_name)",
    )

    # Resume
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Resolve checkpoint directory
    if args.checkpoint_dir is not None:
        pass  # explicit path, use as-is
    elif args.model_name is not None:
        model_dir = models_dir() / args.model_name
        args.checkpoint_dir = str(model_dir / "checkpoints")
    else:
        args.checkpoint_dir = "checkpoints"

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, weights_only=False)
        # Restore args from checkpoint but allow overrides for epochs
        saved_args = ckpt["args"]
        resume_epoch = ckpt["epoch"]
        saved_args["epochs"] = args.epochs
        saved_args["resume"] = args.resume
        saved_args["checkpoint_dir"] = args.checkpoint_dir
        if args.model_name:
            saved_args["model_name"] = args.model_name
        args = argparse.Namespace(**saved_args)
        args._resume_ckpt = ckpt
        args._resume_epoch = resume_epoch
    else:
        args._resume_ckpt = None
        args._resume_epoch = 0

    train(args)

    # Save config.json if using model registry
    model_name = getattr(args, "model_name", None)
    if model_name:
        n_params = sum(
            p.numel() for p in GrassLM(
                vocab_size=30522,
                d_model=args.d_model,
                n_layers=args.n_layers,
                d_reduce=args.d_reduce,
                d_ff=args.d_ff,
                max_seq_len=args.seq_len,
                dropout=0.0,
            ).parameters()
        )
        config = {
            "name": model_name,
            "parameters": n_params,
            "architecture": {
                "vocab_size": 30522,
                "d_model": args.d_model,
                "n_layers": args.n_layers,
                "d_reduce": args.d_reduce,
                "d_ff": args.d_ff,
                "max_seq_len": args.seq_len,
                "window_schedule": [1, 2, 4, 8, 12, 16][:args.n_layers],
            },
            "training": {
                "dataset": "wikitext-2-raw",
                "tokenizer": "bert-base-uncased",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "warmup_steps": args.warmup_steps,
                "seed": args.seed,
            },
            "files": {
                "best_checkpoint": "checkpoints/best_model.pt",
                "latest_checkpoint": "checkpoints/latest.pt",
            },
        }
        save_config(model_name, config)
        print(f"Saved config to models/{model_name}/config.json")
