"""Model i pętla treningowa dla regresji liczby wypożyczeń.

Notebook (`bike-rental-regression.ipynb`) odpowiada za EDA, czyszczenie danych
i budowę DataLoaderów, a następnie importuje stąd model i trening.
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 1000
HIDDEN_SIZE = 20
LEARNING_RATE = 0.01


class Network(nn.Module):
    """Prosty MLP z jedną warstwą ukrytą - regresja wartości `cnt`."""

    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, num_hidden)
        self.act_fn = nn.Tanh()
        self.linear2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        return x


def torch_rmsle(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """RMSLE - metryka, którą oceniane jest zadanie.

    Używana bezpośrednio jako funkcja straty: optymalizujemy dokładnie to,
    co jest potem mierzone. `clamp(min=0)` chroni logarytm przed ujemnymi
    predykcjami sieci.
    """
    msle = torch.mean(
        torch.square(torch.log(torch.clamp(y_pred, min=0) + 1) - torch.log(y_true + 1))
    )
    return torch.sqrt(msle)


def do_train(model, data_loader, optimizer, loss_module, epochs=EPOCHS):
    """Zwraca listę strat z kolejnych kroków (do wykresu zbieżności)."""
    losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)

            Y_pred = model(X).squeeze(dim=1)
            loss = loss_module(Y, Y_pred)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(data_loader):.4f}")

    return losses


def train_model(data_loader, save_plot_to="results/training.png"):
    n_inputs = data_loader.dataset.tensors[0].size()[1]

    model = Network(n_inputs, HIDDEN_SIZE, 1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    losses = do_train(model, data_loader, optimizer, torch_rmsle)

    if save_plot_to:
        fig, ax = plt.subplots(1, layout="constrained")
        ax.plot(losses)
        ax.set_xlabel("krok")
        ax.set_ylabel("RMSLE")
        fig.savefig(save_plot_to)

    return model


@torch.no_grad()
def evaluate(model: nn.Module, data_loader: data.DataLoader) -> torch.Tensor:
    """Predykcje dla całego loadera (działa też, gdy batch nie zawiera Y)."""
    model.eval()
    predictions = [model(batch[0].to(device)).squeeze(dim=1).cpu() for batch in data_loader]
    return torch.cat(predictions)


@torch.no_grad()
def calc_accuracy(model: nn.Module, data_loader: data.DataLoader, loss_module: nn.Module):
    """Wartość metryki na całym loaderze (dla RMSLE - im mniej, tym lepiej)."""
    model.eval()
    predictions, targets = [], []

    for batch in data_loader:
        predictions.append(model(batch[0].to(device)).squeeze(dim=1).cpu())
        targets.append(batch[1].cpu())

    return loss_module(torch.cat(targets), torch.cat(predictions)).item()
