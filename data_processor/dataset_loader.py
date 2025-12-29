import json
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from datasets import DatasetDict

class TranslationDataset(Dataset):
    """
    Loads JSON translation pairs:
    {
        "src": "...",
        "tgt": "..."
    }

    builds GPT-style prompts:
        "Translate English to French:
         English: <src>
         French: "
    """

    def __init__(self,
                 data_path,
                 tokenizer,
                 src_lang="English",
                 tgt_lang="French",
                 max_length=128):
        self.data = json.load(open(data_path))
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_length = max_length

    # prompt template
    def build_prompt(self, text):
        return (
            f"Translate {self.src_lang} to {self.tgt_lang}:\n"
            f"{self.src_lang}: {text}\n"
            f"{self.tgt_lang}:"
        )

    # one sample
    def __getitem__(self, idx):
        example = self.data[idx]

        src = example["src"]
        tgt = example["tgt"]

        # Build translation prompt for GPT-2
        prompt = self.build_prompt(src)

        # encode prompt (input)
        prompt_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).input_ids[0]

        # encode target (label)
        label_ids = self.tokenizer(
            tgt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).input_ids[0]

        return {
            "input_ids": prompt_ids,
            "labels": label_ids
        }

    def __len__(self):
        return len(self.data)


def split_dataset_by_ratio(ds, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Splits a HuggingFace dataset into train/val/test using ratios.

    Args:
        ds (Dataset): HuggingFace dataset (e.g., ds["train"])
        train_ratio (float): ratio for training set
        val_ratio (float): ratio for validation set
        test_ratio (float): ratio for test set
        seed (int): random seed for shuffling

    Returns:
        dict: { "train": train_ds, "val": val_ds, "test": test_ds }
    """

    # 1. Shuffle for reproducibility
    ds = ds.shuffle(seed=seed)

    # 2. Compute absolute sizes
    total = len(ds)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    test_end = total  # remainder

    # 3. Slice dataset
    train_ds = ds.select(range(0, train_end))
    val_ds = ds.select(range(train_end, val_end))
    test_ds = ds.select(range(val_end, test_end))

    return {
        "train": train_ds,
        "val": val_ds,
        "test": test_ds
    }


# collate function for DataLoader
def collate_fn(batch, pad_token_id):
    """
    Pads sequences to same length for a batch.
    GPT2 pad token is usually the EOS token
    """
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100  # ignore loss on padded labels
    )

    return {"input_ids": input_ids, "labels": labels}



# collate function for DataLoader
def collate_fn(batch, pad_token_id):
    """
    Pads sequences to same length for a batch.
    GPT2 pad token is usually the EOS token
    """
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100  # ignore loss on padded labels
    )

    return {"input_ids": input_ids, "labels": labels}
