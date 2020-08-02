import numpy as np
import os

import torch
from torch.utils.data import Dataset


class CreatorDataset(Dataset):
    """
    Use this dataset for the "creator-type" InfoGAN,
    where you want to create new samples from a noise vector z.
    """

    def __init__(self, z_dim=10, c_range=[(0, 5), (0, 5)], discrete=True, dataset_size=1000):
        self.z_dim = z_dim
        self.discrete = discrete
        self.dataset_size = dataset_size

        # limiting to only 2 conditioning variables.
        # frequency and amplitude.
        # if discrete then c_range  = [(low, high)]
        # else c_range  = [(mean, std)]
        self.frequency_range = c_range[0]
        self.amplitude_range = c_range[1]

        if self.discrete:
            self.frequency_func = lambda: np.random.randint(
                self.frequency_range[0], self.frequency_range[1]
            )/self.frequency_range[1]

            self.amplitude_func = lambda: np.random.randint(
                self.amplitude_range[0], self.amplitude_range[1]
            )/self.amplitude_range[1]
        else:
            self.frequency_func = lambda: np.random.normal(
                self.frequency_range[0], self.frequency_range[1]
            )
            self.amplitude_func = lambda: np.random.normal(
                self.amplitude_range[0], self.amplitude_range[1]
            )

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        datapoint_len = np.random.randint(10,50)

        z = torch.normal(0, 1, (datapoint_len, self.z_dim))

        freq = self.frequency_func()
        amplitude = self.amplitude_func()

        c = np.reshape( np.tile([freq, amplitude], datapoint_len), (datapoint_len, 2))
        c = torch.tensor(c,  dtype=torch.float)

        z_c = torch.cat((z,c),1)

        x = np.linspace(-10, 10 , datapoint_len)
        sin = torch.tensor(np.sin(x * freq) + amplitude,  dtype=torch.float).view(-1,1)

        return  z_c, sin


class DiscriminatorDataset(Dataset):
    """
    Feed some widly wrong data into the discriminator,
    use for both creator and converter type.

    Especially useful when using Wasserstein Loss.
    """

    def __init__(self, z_dim=10, c_range=[(0, 10), (0, 10)], discrete=True, dataset_size=1000):
        self.z_dim = z_dim
        self.discrete = discrete
        self.dataset_size = dataset_size

        # limiting to only 2 conditioning variables.
        # frequency and amplitude.
        # if discrete then c_range  = [(low, high)]
        # else c_range  = [(mean, std)]
        self.frequency_range = c_range[0]
        self.amplitude_range = c_range[1]

        if self.discrete:
            self.frequency_func = lambda: np.random.randint(
                self.frequency_range[0], self.frequency_range[1]
            )/self.frequency_range[1]

            self.amplitude_func = lambda: np.random.randint(
                self.amplitude_range[0], self.amplitude_range[1]
            )/self.amplitude_range[1]
        else:
            self.frequency_func = lambda: np.random.normal(
                self.frequency_range[0], self.frequency_range[1]
            )
            self.amplitude_func = lambda: np.random.normal(
                self.amplitude_range[0], self.amplitude_range[1]
            )

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        datapoint_len = np.random.randint(10,50)

        z = torch.normal(0, 1, (datapoint_len, self.z_dim ,1))

        freq = self.frequency_func()
        amplitude = self.amplitude_func()

        c = np.reshape( np.tile([freq, amplitude], datapoint_len), (datapoint_len, 2, 1))
        c = torch.tensor(c,  dtype=torch.float)

        z_c = torch.cat((z,c),1)

        x = np.linspace(-10, 10 , datapoint_len)
        sin = torch.tensor(np.sin(x * freq) + amplitude,  dtype=torch.float).view(-1,1)

        return  z_c, sin
