"""Wczytanie, czyszczenie i tokenizacja komentarzy."""

import re

import numpy as np
import pandas as pd
import torch
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from config import BATCH_SIZE, MODEL_NAME, SEED, TEST_PATH, TRAIN_PATH, VAL_SIZE, device


def preprocess_text(text):
    """Encje HTML, linki i @nicki zamieniamy na stałe tokeny.

    Komentarze pochodzą z portali i Twittera - bez tego model uczyłby się
    konkretnych adresów i nazw użytkowników zamiast języka.
    """
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)
    text = re.sub(r"@\w+", "[USER]", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def pick_max_length(texts, tokenizer, cap=128):
    """MAX_LENGTH = 95. percentyl długości, zaokrąglony w górę do wielokrotności 8.

    Stała 128 marnowałaby obliczenia - komentarze są krótkie (p95 ~ 38 tokenów).
    """
    token_lengths = [len(tokenizer(t)["input_ids"]) for t in texts]
    p95 = int(np.percentile(token_lengths, 95))
    return int(min(cap, max(32, (p95 + 7) // 8 * 8)))


def load_data(batch_size=BATCH_SIZE):
    """Zwraca (train_loader, val_loader, class_weights, tokenizer, max_length)."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    df = pd.read_csv(TRAIN_PATH)
    df["sentence"] = df["sentence"].apply(preprocess_text)
    texts = df["sentence"].tolist()
    labels = df["label"].astype(int).tolist()

    max_length = pick_max_length(texts, tokenizer)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=VAL_SIZE, random_state=SEED, stratify=labels
    )

    train_ds = HateSpeechDataset(train_texts, train_labels, tokenizer, max_length)
    val_ds = HateSpeechDataset(val_texts, val_labels, tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ~8.5% próbek to hejt - bez ważenia model uczy się przewidywać samo 0.
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(train_labels), y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    return train_loader, val_loader, class_weights, tokenizer, max_length


def load_test_texts():
    with open(TEST_PATH, encoding="utf-8") as f:
        raw_lines = f.read().splitlines()
    return [preprocess_text(line) for line in raw_lines]
