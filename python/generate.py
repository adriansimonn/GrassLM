"""
Text generation for GrassLM: greedy, top-k, and top-p (nucleus) sampling.

Command:
    python generate.py --checkpoint checkpoints/best_model.pt \
                       --prompt "The meaning of" \
                       [--max_tokens 100] [--temperature 1.0] \
                       [--top_k 0] [--top_p 0.9] [--greedy]
"""

import argparse

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from grasslm.model import GrassLM


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[GrassLM, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = ckpt["args"]

    model = GrassLM(
        vocab_size=saved_args.get("vocab_size", 30522),
        d_model=saved_args["d_model"],
        n_layers=saved_args["n_layers"],
        d_reduce=saved_args["d_reduce"],
        d_ff=saved_args["d_ff"],
        max_seq_len=saved_args["seq_len"],
        dropout=0.0,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, saved_args


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    # Zero out all logits outside the top-k entries.
    if k <= 0:
        return logits
    top_k_vals, _ = torch.topk(logits, k)
    threshold = top_k_vals[:, -1].unsqueeze(-1)
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    # Nucleus sampling: keep the smallest set of tokens with cumulative prob >= p.
    if p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    # Shift right so the first token above threshold is kept
    sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= p
    sorted_logits[sorted_mask] = float("-inf")

    # Scatter back to original ordering
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)


@torch.no_grad()
def generate(
    model: GrassLM,
    input_ids: torch.Tensor,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    greedy: bool = False,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """
    Autoregressively generate tokens.

    Args:
        model: GrassLM model in eval mode.
        input_ids: Prompt token IDs, shape (1, prompt_len).
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (ignored if greedy).
        top_k: Top-k filtering (0 = disabled).
        top_p: Nucleus sampling threshold (1.0 = disabled).
        greedy: If True, always pick argmax.
        eos_token_id: Stop generation if this token is produced.

    Returns:
        Full sequence including prompt, shape (1, prompt_len + generated_len).
    """
    max_seq_len = model.max_seq_len
    generated = input_ids

    for _ in range(max_tokens):
        # Truncate to max_seq_len if needed (keep most recent tokens)
        if generated.size(1) >= max_seq_len:
            context = generated[:, -max_seq_len:]
        else:
            context = generated

        logits = model(context)  # (1, context_len, V)
        next_logits = logits[:, -1, :]  # (1, V)

        if greedy:
            next_token = next_logits.argmax(dim=-1, keepdim=True)
        else:
            # Apply temperature
            if temperature != 1.0:
                next_logits = next_logits / temperature

            # Apply top-k filtering
            next_logits = top_k_filter(next_logits, top_k)

            # Apply top-p (nucleus) filtering
            next_logits = top_p_filter(next_logits, top_p)

            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=1)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with GrassLM")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pt)",
    )
    parser.add_argument(
        "--prompt", type=str, default="The meaning of",
        help="Text prompt to continue from",
    )
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0,
                        help="Top-k filtering (0 = disabled)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus sampling threshold (1.0 = disabled)")
    parser.add_argument("--greedy", action="store_true",
                        help="Use greedy decoding (argmax)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model, saved_args = load_model_from_checkpoint(args.checkpoint, device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Tokenize prompt
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    mode = "greedy" if args.greedy else f"sampling (T={args.temperature}, top_k={args.top_k}, top_p={args.top_p})"
    print(f"Prompt: \"{args.prompt}\"")
    print(f"Mode: {mode}")
    print(f"Max tokens: {args.max_tokens}")
    print("-" * 60)

    # Generate
    output_ids = generate(
        model,
        input_ids,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        greedy=args.greedy,
        eos_token_id=tokenizer.sep_token_id,
    )

    # Decode full output
    output_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
    print(output_text)


if __name__ == "__main__":
    main()
