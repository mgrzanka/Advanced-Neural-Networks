from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import pickle
import os
import random

PAD_VALUE = 193
PAUSE_INDEX = 192

class MusicAcordsDataset(Dataset):
    def __init__(self, data, has_targets=True, use_embedding=True):
        super().__init__()
        self.sequences = []
        self.classes = []
        self.has_targets = has_targets
        self.use_embedding = use_embedding

        for sample in data:
            if self.has_targets:
                seq = sample[0]
                self.classes.append(torch.tensor(sample[1], dtype=torch.long))
            else:
                seq = sample

            seq_tensor = torch.tensor(seq, dtype=torch.float32)
            seq_tensor = torch.nan_to_num(seq_tensor, nan=float(PAD_VALUE))
            seq_tensor = seq_tensor.to(torch.long)
            seq_tensor[seq_tensor < 0] = PAUSE_INDEX
            seq_tensor[seq_tensor > PAD_VALUE] = PAD_VALUE

            if self.use_embedding:
                self.sequences.append(seq_tensor) # [seq_length]
            else:
                self.sequences.append(seq_tensor.to(torch.float32).unsqueeze(1)) # [seq_length, 1]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, indx):
        if self.has_targets:
            return self.sequences[indx], self.classes[indx]
        return self.sequences[indx], None

def pad_collate(batch):
    xx = [item[0] for item in batch]
    yy = [item[1] for item in batch if item[1] is not None]

    x_lens = torch.tensor([len(x) for x in xx], dtype=torch.long)
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=PAD_VALUE)

    if len(yy) > 0:
        yy_stack = torch.stack(yy)
        return xx_pad, yy_stack, x_lens

    return xx_pad, None, x_lens

def create_dataset(batch_size, use_embedding=True):
    with open(os.path.join("data", "test_no_target.pkl"), 'rb') as f:
        test_data = pickle.load(f)

    with open(os.path.join("data", "train.pkl"), 'rb') as f:
        data = pickle.load(f)

    random.seed(42)
    random.shuffle(data)

    train_size = int(0.7 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]

    train_set = MusicAcordsDataset(train_data, has_targets=True, use_embedding=use_embedding)
    val_set = MusicAcordsDataset(val_data, has_targets=True, use_embedding=use_embedding)
    test_set = MusicAcordsDataset(test_data, has_targets=False, use_embedding=use_embedding)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=pad_collate)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=pad_collate)

    return train_loader, val_loader, test_loader
