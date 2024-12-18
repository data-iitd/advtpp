import numpy as np
import pdb
import torch
import torch.utils.data
from transformer import Constants

class EventData(torch.utils.data.Dataset):
    def __init__(self, data):
        self.time = [[elem['time_since_start'] for elem in inst] for inst in data]
        self.time_gap = [[elem['time_since_last_event'] for elem in inst] for inst in data]
        self.event_type = [[elem['type_event'] + 1 for elem in inst] for inst in data]
        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.time[idx], self.time_gap[idx], self.event_type[idx]

def pad_time(insts, max_len):
    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.float32)

def pad_type(insts, max_len):
    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.long)


class Collator(object):
    def __init__(self, pad_max_len):
        self.pad_max_len = pad_max_len

    def __call__(self, insts):
        time, time_gap, event_type = list(zip(*insts))
        time = pad_time(time, self.pad_max_len)
        time_gap = pad_time(time_gap, self.pad_max_len)
        event_type = pad_type(event_type, self.pad_max_len)

        return time, time_gap, event_type

def get_dataloader(data, batch_size, pad_max_len, shuffle=True):
    ds = EventData(data)
    collator = Collator(pad_max_len)

    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=2,
        batch_size=batch_size,
        collate_fn=collator,
        shuffle=shuffle
    )
    return dl
