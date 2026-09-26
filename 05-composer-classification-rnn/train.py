from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch
import os


def train(epochs, model, train_loader, val_loader):
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-6
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    best_val_loss = float('inf')
    save_path = 'best_model.pth'

    for epoch in range(epochs):
        # TRAINING
        model.train()
        train_epoch_loss = 0.0

        for x, targets, x_len in train_loader:
            x, targets = x.to(device), targets.to(device)

            optimizer.zero_grad()

            preds = model(x, x_len)

            loss = loss_func(preds, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_epoch_loss += loss.item()

        # VALIDATION
        model.eval()

        all_preds = []
        all_targets = []
        val_loss_total = 0.0

        for x, targets, x_len in val_loader:
            x, targets = x.to(device), targets.to(device)

            with torch.no_grad():
                logits = model(x, x_len)

                predicted_classes = torch.argmax(logits, dim=1)

                all_preds.append(predicted_classes)
                all_targets.append(targets)
                loss = loss_func(logits, targets)
                val_loss_total += loss.item()

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        avg_train_loss = train_epoch_loss / len(train_loader)
        avg_val_loss = val_loss_total / len(val_loader)
        val_acc = torch.sum(all_preds == all_targets).float() / len(all_preds)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            saved_info = f"| The best model saved"
        else:
            saved_info = ""

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch: {epoch:3d} | LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}{saved_info}")
