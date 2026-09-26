from dataset import create_dataset, PAD_VALUE
from models import create_model
from train import train
from sample import sample

EPOCHS = 100
BATCH_SIZE = 32
LSTM_HIDDEN_SIZE = 64
CLASSIFIER_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
MEAN_POOLING = True
BIDIRECTIONAL = True

USE_EMBEDDING = True
EMBEDDING_DIM = 32
VOCAB_SIZE = 194  # Acords (0-191) + Pause (192) + Padding (193)


if __name__ == '__main__':
    print(f"Experiment: {EPOCHS=}, {BATCH_SIZE=}, {LSTM_HIDDEN_SIZE=}, {CLASSIFIER_HIDDEN_SIZE=}, {LSTM_NUM_LAYERS=}, {MEAN_POOLING=}, {BIDIRECTIONAL=}, {USE_EMBEDDING=}")

    train_loader, val_loader, test_loader = create_dataset(BATCH_SIZE, USE_EMBEDDING)

    model = create_model(
        lstm_hidden_size=LSTM_HIDDEN_SIZE,
        lstm_num_layers=LSTM_NUM_LAYERS,
        lstm_is_bidirectional=BIDIRECTIONAL,
        classifier_hidden_size=CLASSIFIER_HIDDEN_SIZE,
        classifier_use_mean_pooling=MEAN_POOLING,
        use_embedding=USE_EMBEDDING,
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        pad_value=PAD_VALUE
    )

    print("Training started")
    train(EPOCHS, model, train_loader, val_loader)

    print("Sampling started")
    sample(model, test_loader)
