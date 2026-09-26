from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch


class LSTM_Seq_Regressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        dropout_prob = 0.5 if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            bidirectional = bidirectional,
            dropout = dropout_prob
        )

    def forward(self, x):
        all_outputs, (h_n, c_n) = self.lstm(x)
        return all_outputs, (h_n, c_n)


class MusicAuthorClassifier(nn.Module):
    def __init__(self, seq_model: LSTM_Seq_Regressor, num_classes, hidden_size, use_mean_pooling, bidirectional, use_embedding, vocab_size, embedding_dim, pad_value):
        super().__init__()
        self.use_mean_pooling = use_mean_pooling
        self.seq_model = seq_model
        self.bidirectional = bidirectional
        self.use_embedding = use_embedding

        if self.use_embedding:
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim,
                padding_idx=pad_value
            )

        input_size = seq_model.hidden_size * 2 if bidirectional else seq_model.hidden_size

        self.classification_head = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Dropout(p=0.5),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x, x_len):
        if self.use_embedding:
            x = self.embedding(x)

        x_packed = pack_padded_sequence(x, x_len.cpu(), batch_first=True, enforce_sorted=False)

        all_outputs, (h_n, c_n) = self.seq_model(x_packed)

        if self.use_mean_pooling:
            unpacked_out, _ = pad_packed_sequence(all_outputs, batch_first=True) # [batch, seq, hidden_size]

            sum_hidden = torch.sum(unpacked_out, dim=1)
            pooled_hidden = sum_hidden / x_len.unsqueeze(1).float().to(sum_hidden.device)

            out = self.classification_head(pooled_hidden)

        elif self.bidirectional:
            forward_hidden = h_n[-2]
            backward_hidden = h_n[-1]
            last_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)   # [batch, seq, 2*hidden_size]
            out = self.classification_head(last_hidden)

        else:
            last_hidden = h_n[-1] # [batch, seq, hidden_size]
            out = self.classification_head(last_hidden)

        return out


def create_model(lstm_hidden_size, lstm_num_layers, lstm_is_bidirectional, classifier_hidden_size, classifier_use_mean_pooling, use_embedding, vocab_size, embedding_dim, pad_value):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lstm_input_size = embedding_dim if use_embedding else 1

    lstm_base = LSTM_Seq_Regressor(
        input_size=lstm_input_size,
        hidden_size=lstm_hidden_size,
        num_layers=lstm_num_layers,
        bidirectional=lstm_is_bidirectional
    )

    model = MusicAuthorClassifier(
        seq_model=lstm_base,
        num_classes=5,
        hidden_size=classifier_hidden_size,
        use_mean_pooling=classifier_use_mean_pooling,
        bidirectional=lstm_is_bidirectional,
        use_embedding=use_embedding,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        pad_value=pad_value
    ).to(device)

    return model
