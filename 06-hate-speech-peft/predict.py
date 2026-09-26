"""Predykcje na zbiorze testowym -> pred.csv (bez nagłówka i indeksu)."""

import pandas as pd
from torch.utils.data import DataLoader

from data import HateSpeechDataset, load_test_texts


def save_predictions(model, tokenizer, max_length, threshold=0.5,
                     output_path="pred.csv", batch_size=32):
    from train import predict_proba

    test_texts = load_test_texts()
    print(f"Test samples: {len(test_texts)}")

    test_ds = HateSpeechDataset(test_texts, None, tokenizer, max_length)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Próg dobrany na walidacji zamiast surowego argmax.
    test_probs, _ = predict_proba(model, test_loader)
    test_preds = (test_probs >= threshold).astype(int)

    pd.Series(test_preds, dtype=int).to_csv(output_path, index=False, header=False)
    print(f"Saved {output_path}: {len(test_preds)} rows (threshold={threshold:.2f})")

    return test_preds
