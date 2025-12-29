# this file will contain helpers for loading and cleaning our dataset
from datasets import load_dataset

import json
import torch
import random

from transformers import AutoTokenizer
from dictionary import DictionaryHelper


class TranslationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        tokenizer,
        max_length=512,
        hint_dropout = 0.4, # drop hints 40% of the time
        src_lang="English",
        tgt_lang="Cantonese",
    ):
        self.data = json.load(open(data_path))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.hint_dropout = hint_dropout
        self.src = src_lang
        self.tgt = tgt_lang

        self.dictionary = DictionaryHelper()

    def build_prompt_with_hints(self, text):
        drop_hint = random.random() < self.hint_dropout
        hints = "" if drop_hint else self.dictionary.get_hints(text)
        return f"{hints}{self.build_prompt(text)}"

    def build_prompt(self, text):
        return f"translate {self.src} to {self.tgt}: {text}"

    def __getitem__(self, idx):
        example = self.data[idx]

        target_text = example["tgt"]

        if not target_text or not target_text.strip():
            target_text = "<empty>"

        # seq2seq: encode targets and inputs separately
        prompt = self.build_prompt_with_hints(example["src"])
        inputs_encoded = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # using text_target for correct Seq2Seq tokenization
        labels_encoded = self.tokenizer(
            text_target=target_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        label_ids = labels_encoded["input_ids"].squeeze(0)

        # pading tokens
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs_encoded["input_ids"].squeeze(0),
            "attention_mask": inputs_encoded["attention_mask"].squeeze(0),
            "labels": label_ids,
        }

    def __len__(self):
        return len(self.data)
