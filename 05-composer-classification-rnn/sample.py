import torch
import pandas as pd
import os

def sample(model, test_loader, weights_path='best_model.pth'):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Pomyślnie załadowano wagi z pliku: {weights_path}")
    else:
        raise ValueError("Weights path not found")

    model.eval()
    all_preds = []

    with torch.no_grad():
        for x, _, x_len in test_loader:
            x = x.to(device)
            logits = model(x, x_len)
            predicted_classes = torch.argmax(logits, dim=1)
            all_preds.extend(predicted_classes.cpu().numpy())

    df = pd.DataFrame(data=all_preds, columns=["predicted_class"])
    df.to_csv("pred.csv", index=False, header=False)
    print("Preds saved")
