from torch import nn
import torch


class NeuralNetwork(nn.Module):
    def __init__(self, output_size, network_sizes, input_size_cont, emb_dims=None, p=None, use_batch_norm=True):
        super(NeuralNetwork, self).__init__()

        self.emb_layers = nn.ModuleList()
        total_emb_size = 0
        if emb_dims is not None:
            for num_classes, emb_dim in emb_dims:
                self.emb_layers.append(nn.Embedding(num_classes, emb_dim))
                total_emb_size += emb_dim
        self.act_emb = nn.Tanh()

        total_input_size = input_size_cont + total_emb_size

        layers = []
        layers.append(nn.Linear(total_input_size, network_sizes[0]))
        layers.append(nn.ReLU())

        size_in = network_sizes[0]
        for size in network_sizes[1:]:
             layers.append(nn.Linear(size_in, size))
             if use_batch_norm:
                layers.append(nn.BatchNorm1d(size))
             layers.append(nn.ReLU())
             if p is not None:
                  layers.append(nn.Dropout(p))
             size_in = size

        if use_batch_norm:
            layers.append(nn.BatchNorm1d(size_in))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(size_in, output_size))

        self.layers = nn.Sequential(*layers)

    def forward(self, X_cont: torch.Tensor, X_cat: torch.Tensor | None = None):
        embeddings = []
        if X_cat is not None and len(self.emb_layers) > 0:
            for i, emb_layer in enumerate(self.emb_layers):
                embedded_cat_col = self.act_emb(emb_layer(X_cat[:, i].long()))
                embeddings.append(embedded_cat_col)
        if embeddings:
            x_cat_embedded = torch.cat(embeddings, dim=1)
            X_combined = torch.cat([X_cont, x_cat_embedded], dim=1)
        else:
            X_combined = X_cont

        output = self.layers(X_combined)
        return output
