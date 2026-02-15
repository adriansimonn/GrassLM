"""
Evaluate GrassLM perplexity on Wikitext-2 test set.

Command:
    python evaluate.py --model_name GrassLM-10M [--split test]
                       [--batch_size 32] [--seq_len 128]

    # Or with explicit checkpoint path:
    python evaluate.py --checkpoint path/to/best_model.pt [--split test]
"""

import argparse
import math

import torch
import torch.nn as nn

from grasslm.data import create_dataloaders
from grasslm.model import GrassLM
from grasslm.registry import resolve_checkpoint


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def compute_perplexity(
    model: nn.Module,
    loader,
    device: torch.device,
) -> tuple[float, float]:
    # Returns (perplexity, avg_loss).
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids)
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        total_loss += loss.item()
        total_tokens += labels.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity, avg_loss


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[GrassLM, dict]:
    # Load a GrassLM model from a training checkpoint.
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = ckpt["args"]

    model = GrassLM(
        vocab_size=saved_args.get("vocab_size", 30522),
        d_model=saved_args["d_model"],
        n_layers=saved_args["n_layers"],
        d_reduce=saved_args["d_reduce"],
        d_ff=saved_args["d_ff"],
        max_seq_len=saved_args["seq_len"],
        dropout=0.0,  # no dropout at eval
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, saved_args


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate GrassLM perplexity on Wikitext-2"
    )
    parser.add_argument(
        "--model_name", type=str, default=None,
        help="Model name (e.g. GrassLM-10M). Loads best checkpoint from models/<name>/",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (.pt). Overrides --model_name.",
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["val", "test"],
        help="Which split to evaluate on (default: test)",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=None,
                        help="Override seq_len (default: use checkpoint value)")
    args = parser.parse_args()

    # Resolve checkpoint path
    checkpoint = args.checkpoint
    if checkpoint is None and args.model_name is not None:
        checkpoint = resolve_checkpoint(args.model_name)
    if checkpoint is None:
        parser.error("Either --model_name or --checkpoint is required")

    device = get_device()
    print(f"Using device: {device}")

    # Load model
    print(f"Loading checkpoint: {checkpoint}")
    model, saved_args = load_model_from_checkpoint(checkpoint, device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {saved_args['n_layers']}L, d={saved_args['d_model']}, "
          f"params={n_params:,}")

    # Load data
    seq_len = args.seq_len or saved_args["seq_len"]
    train_loader, val_loader, test_loader = create_dataloaders(
        seq_len=seq_len,
        batch_size=args.batch_size,
    )

    loader = test_loader if args.split == "test" else val_loader
    print(f"Evaluating on {args.split} split ({len(loader)} batches)...")

    # Evaluate
    ppl, avg_loss = compute_perplexity(model, loader, device)

    print(f"\nResults ({args.split} set):")
    print(f"  Average loss: {avg_loss:.4f}")
    print(f"  Perplexity:   {ppl:.2f}")


if __name__ == "__main__":
    main()
