"""Trening i porównanie czterech metod dostrajania.

Uruchomienie:
    python train.py

Skrypt trenuje kolejno transfer / full / lora / soft_prompt, wybiera wariant
o najlepszym macro-F1 na walidacji, dobiera próg decyzyjny i zapisuje predykcje.
"""

import gc
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import get_linear_schedule_with_warmup

from config import EPOCHS, LR, METHOD_CONFIGS, WARMUP_RATIO, WEIGHT_DECAY, device, set_seed
from data import load_data
from model import METHODS, build_model


@torch.no_grad()
def evaluate(model, loader, criterion=None):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        if criterion is not None:
            total_loss += criterion(logits, labels).item()

        all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader) if criterion is not None else None
    # macro-F1, bo przy 8.5% klasy pozytywnej accuracy niczego nie mówi.
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1, np.array(all_preds), np.array(all_labels)


@torch.no_grad()
def predict_proba(model, loader):
    """Prawdopodobieństwa klasy 'hejt' (+ etykiety, jeśli loader je zawiera)."""
    model.eval()
    probs, labels_out = [], []
    has_labels = False

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        if "labels" in batch:
            labels_out.extend(batch["labels"].numpy())
            has_labels = True

    return np.array(probs), (np.array(labels_out) if has_labels else None)


def train_model(model, train_loader, val_loader, class_weights,
                epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY):
    model.to(device)
    # Tylko odmrożone parametry trafiają do optymalizatora - istotne dla PEFT.
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * WARMUP_RATIO), total_steps
    )

    best_f1 = -1.0
    best_state = None
    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    for epoch in range(epochs):
        model.train()
        running = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()

        train_loss = running / len(train_loader)
        val_loss, val_f1, _, _ = evaluate(model, val_loader, criterion)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        print(f"Epoch {epoch + 1}/{epochs} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_macroF1={val_f1:.4f}")

        # Zapisujemy wagi z najlepszej epoki, nie z ostatniej.
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = deepcopy(model.state_dict())
            print(f"  New best macro-F1: {best_f1:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"Best macro-F1 (validation): {best_f1:.4f}")
    return model, history


def tune_threshold(model, val_loader):
    """Przy niezbalansowanych klasach próg 0.5 rzadko maksymalizuje macro-F1.

    Próg dobierany jest WYŁĄCZNIE na walidacji i dopiero potem stosowany do testu.
    """
    val_probs, val_labels = predict_proba(model, val_loader)
    thresholds = np.linspace(0.05, 0.95, 91)
    f1_by_t = [
        f1_score(val_labels, (val_probs >= t).astype(int), average="macro", zero_division=0)
        for t in thresholds
    ]
    best_idx = int(np.argmax(f1_by_t))
    best_threshold = float(thresholds[best_idx])

    f1_default = f1_score(val_labels, (val_probs >= 0.5).astype(int),
                          average="macro", zero_division=0)
    print(f"Default threshold 0.50 -> macro-F1 = {f1_default:.4f}")
    print(f"Tuned threshold {best_threshold:.2f} -> macro-F1 = {f1_by_t[best_idx]:.4f} "
          f"(gain: {f1_by_t[best_idx] - f1_default:+.4f})")

    return best_threshold


def clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_comparison(methods=METHODS):
    train_loader, val_loader, class_weights, tokenizer, max_length = load_data()

    results = {}
    best_method, best_f1, best_model = None, -1.0, None

    for method in methods:
        cfg = METHOD_CONFIGS[method]
        print(f"\n========== Method: {method} (lr={cfg['lr']}, epochs={cfg['epochs']}) ==========")
        set_seed()

        candidate = build_model(method)
        candidate, _ = train_model(
            candidate, train_loader, val_loader, class_weights,
            epochs=cfg["epochs"], lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        )

        _, f1, _, _ = evaluate(candidate, val_loader)
        results[method] = f1
        print(f"Method {method}: validation macro-F1 = {f1:.4f}")

        if f1 > best_f1:
            best_f1, best_method = f1, method
            if best_model is not None:
                del best_model
            best_model = candidate
        else:
            del candidate

        clear_vram()

    print("\n========== Summary (macro-F1 on validation) ==========")
    for meth, f1 in sorted(results.items(), key=lambda kv: kv[1], reverse=True):
        mark = "   <-- best" if meth == best_method else ""
        print(f"{meth:12s}: {f1:.4f}{mark}")

    _, val_f1, val_preds, val_true = evaluate(best_model, val_loader)
    print(f"\nBest method: {best_method}")
    print(f"Macro-F1: {val_f1:.4f} | Accuracy: {accuracy_score(val_true, val_preds):.4f}\n")
    print(classification_report(val_true, val_preds,
                                target_names=["Non-hate (0)", "Hate (1)"], zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(val_true, val_preds))

    threshold = tune_threshold(best_model, val_loader)
    return best_model, tokenizer, max_length, threshold, results


if __name__ == "__main__":
    from predict import save_predictions

    model, tokenizer, max_length, threshold, _ = run_comparison()
    save_predictions(model, tokenizer, max_length, threshold)
