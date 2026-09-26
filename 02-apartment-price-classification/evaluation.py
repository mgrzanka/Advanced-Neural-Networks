import numpy as np
import pandas as pd
import torch


def calc_accuracy_training(out: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(out, dim=1)

    accuracies = []
    for i in range(3):
        class_mask = (labels == i)

        if class_mask.sum() > 0:
            class_correct = (preds[class_mask] == labels[class_mask]).sum().float()
            class_acc = class_correct / class_mask.sum().float()
            accuracies.append(class_acc.item())

    if not accuracies:
        return 0.0
    return np.mean(accuracies, dtype=float)


def calc_accuracy(pred_targets, targets):
    accuracies = []
    for i in range(3):
        class_correct=(pred_targets == targets.values)[targets == i].sum()
        accuracies.append(class_correct/(targets == i).sum())
    return(np.mean(accuracies))


if __name__ == '__main__':
    predictions_student = pd.read_csv("pred.csv", header=None).iloc[:,0]
    labels = pd.read_csv("UNKNOWN.csv", header=None).iloc[:,0]

    print(calc_accuracy(predictions_student, labels))
