"""Wspólna konfiguracja dla klasyfikacji mowy nienawiści."""

import random

import numpy as np
import torch


MODEL_NAME = "allegro/herbert-base-cased"  # polski BERT od Allegro
SEED = 42
VAL_SIZE = 0.2
BATCH_SIZE = 16
EPOCHS = 4
LR = 2e-5
WEIGHT_DECAY = 0.01
DROPOUT = 0.1
CLASSIFIER_DROPOUT = 0.2
WARMUP_RATIO = 0.1
MAX_LENGTH = 128  # nadpisywane percentylem długości w data.py

TRAIN_PATH = "data/hate_train.csv"
TEST_PATH = "data/hate_test_data.txt"

# Hiperparametry dobrane osobno per metoda - PEFT znosi (i potrzebuje)
# znacznie większego LR niż pełny fine-tuning.
METHOD_CONFIGS = {
    "transfer":    dict(lr=1e-3, epochs=6,  weight_decay=0.01),
    "full":        dict(lr=2e-5, epochs=4,  weight_decay=0.01),
    "lora":        dict(lr=1e-4, epochs=5,  weight_decay=0.01),
    "soft_prompt": dict(lr=1e-2, epochs=12, weight_decay=0.0),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
