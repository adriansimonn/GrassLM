"""
Data pipeline for GrassLM: Wikitext loading, tokenization, and chunking.

Loads Wikitext-2-raw or Wikitext-103-raw via HuggingFace datasets, tokenizes
with the BERT WordPiece tokenizer (bert-base-uncased, vocab=30522), chunks into
fixed-length blocks, and returns input_ids/labels pairs for causal language
modeling.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


class WikitextCausalLMDataset(Dataset):
    """
    Fixed-length causal LM dataset built from Wikitext-2-raw.

    Concatenates all tokenized text into a single stream, then slices it into
    non-overlapping blocks of (seq_len + 1) tokens. Each sample yields:
        input_ids: block[:-1]  (length seq_len)
        labels:    block[1:]   (length seq_len)
    """

    def __init__(self, token_ids: torch.Tensor, seq_len: int = 128):
        """
        Args:
            token_ids: 1-D tensor of token IDs (full split concatenated).
            seq_len: Context length for each training sample.
        """
        self.seq_len = seq_len
        # We need seq_len + 1 tokens per sample (input + 1 shifted label)
        n_tokens = token_ids.size(0)
        n_samples = n_tokens // (seq_len + 1)
        # Trim to exact multiple
        self.data = token_ids[: n_samples * (seq_len + 1)].view(n_samples, seq_len + 1)

    def __len__(self) -> int:
        return self.data.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        block = self.data[idx]
        return {
            "input_ids": block[:-1],   # (seq_len,)
            "labels": block[1:],       # (seq_len,)
        }


def tokenize_split(
    split_text: list[str],
    tokenizer: AutoTokenizer,
) -> torch.Tensor:
    """
    Tokenize a list of text strings and concatenate into a single 1-D tensor.

    Filters out empty lines, tokenizes without special tokens (no [CLS]/[SEP]),
    and concatenates all token IDs into one flat stream.

    Args:
        split_text: List of raw text strings from the dataset split.
        tokenizer: HuggingFace tokenizer instance.

    Returns:
        1-D LongTensor of all token IDs concatenated.
    """
    all_ids = []
    for text in split_text:
        text = text.strip()
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(ids)
    return torch.tensor(all_ids, dtype=torch.long)


DATASETS = {
    "wikitext-2-raw": ("wikitext", "wikitext-2-raw-v1"),
    "wikitext-103-raw": ("wikitext", "wikitext-103-raw-v1"),
}


def load_wikitext(
    seq_len: int = 128,
    tokenizer_name: str = "bert-base-uncased",
    dataset_name: str = "wikitext-2-raw",
) -> tuple[WikitextCausalLMDataset, WikitextCausalLMDataset, WikitextCausalLMDataset]:
    """
    Load and prepare a Wikitext dataset for causal LM training.

    Args:
        seq_len: Context length for each training sample.
        tokenizer_name: HuggingFace tokenizer to use.
        dataset_name: Which dataset to load ("wikitext-2-raw" or "wikitext-103-raw").

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Available: {list(DATASETS.keys())}"
        )
    hf_path, hf_name = DATASETS[dataset_name]
    dataset = load_dataset(hf_path, hf_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    train_ids = tokenize_split(dataset["train"]["text"], tokenizer)
    val_ids = tokenize_split(dataset["validation"]["text"], tokenizer)
    test_ids = tokenize_split(dataset["test"]["text"], tokenizer)

    train_dataset = WikitextCausalLMDataset(train_ids, seq_len=seq_len)
    val_dataset = WikitextCausalLMDataset(val_ids, seq_len=seq_len)
    test_dataset = WikitextCausalLMDataset(test_ids, seq_len=seq_len)

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    seq_len: int = 128,
    batch_size: int = 32,
    tokenizer_name: str = "bert-base-uncased",
    num_workers: int = 0,
    dataset_name: str = "wikitext-2-raw",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train/val/test splits.

    Args:
        seq_len: Context length for each training sample.
        batch_size: Batch size for all splits.
        tokenizer_name: HuggingFace tokenizer to use.
        num_workers: Number of dataloader worker processes.
        dataset_name: Which dataset to load ("wikitext-2-raw" or "wikitext-103-raw").

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_ds, val_ds, test_ds = load_wikitext(
        seq_len=seq_len, tokenizer_name=tokenizer_name, dataset_name=dataset_name
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
