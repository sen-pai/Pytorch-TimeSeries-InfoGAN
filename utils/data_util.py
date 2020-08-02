import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def pad_collate_creator(batch):
    """
    collate function that pads with zeros for variable lenght data-points.
    pass into the dataloader object.
    """
    z_c, sin = zip(*batch)
    lens = [x.shape[0] for x in z_c]

    z_c = pad_sequence(z_c, batch_first=True, padding_value=0)
    sin = pad_sequence(sin, batch_first=True, padding_value=0)

    return z_c, sin, torch.tensor(lens, dtype=torch.float)
