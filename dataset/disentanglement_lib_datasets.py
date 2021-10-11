import os
import numpy as np
import torch

from random import getrandbits
from PIL import Image
from torch.utils.data import Dataset


class BiasedDisentanglementLibDataset(Dataset):
    """
    Data-loading from Disentanglement Library

    Note:
        Unlike a traditional Pytorch dataset, indexing with _any_ index fetches a random batch.
        What this means is dataset[0] != dataset[0]. Also, you'll need to specify the size
        of the dataset, which defines the length of one training epoch.

        This is done to ensure compatibility with disentanglement_lib.
    """

    def __init__(self, name, transform, target_attr: int, bias_attr: int, bias_degree: float, seed=0):
        """
        Parameters
        ----------
        name : str
            Name of the dataset use. You may use `get_dataset_name`.
        seed : int
            Random seed.
        iterator_len : int
            Length of the dataset. This defines the length of one training epoch.
        """
        assert name == 'smallnorb'
        assert bias_attr != target_attr
        assert transform is not None
        self.name = name
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        from disentanglement_lib.data.ground_truth.named_data import get_named_ground_truth_data
        self.dataset = get_named_ground_truth_data(self.name)
        self.iterator_len = self.dataset.images.shape[0]
        self.target_attr = target_attr
        self.bias_attr = bias_attr
        target_factor_index = self.dataset.latent_factor_indices[self.target_attr]
        bias_factor_index = self.dataset.latent_factor_indices[self.bias_attr]
        self.target_factor_size = self.dataset.factor_sizes[target_factor_index]
        self.bias_factor_size = self.dataset.factor_sizes[bias_factor_index]
        self.transform = transform
        self.bias_attr_val_prob_conditioned_on_target_val = [[bias_degree, 1 - bias_degree], [1 - bias_degree, bias_degree]]

    @staticmethod
    def has_labels():
        return False

    def num_channels(self):
        return self.dataset.observation_shape[2]

    def __len__(self):
        return self.iterator_len

    def __getitem__(self, item):
        assert item < self.iterator_len

        factors = self.dataset.sample_factors(1, random_state=self.random_state)
        # randomly sample target attribtue value
        target_attr_binary_val = getrandbits(1)
        biasd_attr_binary_val = np.random.choice([0, 1],
                                                 p=self.bias_attr_val_prob_conditioned_on_target_val[target_attr_binary_val])

        half_target_size = int(self.target_factor_size / 2)
        half_bias_size = int(self.bias_factor_size / 2)
        if target_attr_binary_val:
            target_factor_val = np.random.randint(0, half_target_size)
        else:
            target_factor_val = np.random.randint(half_target_size, self.target_factor_size)

        if biasd_attr_binary_val:
            bias_factor_val = np.random.randint(0, half_bias_size)
        else:
            bias_factor_val = np.random.randint(half_bias_size, self.bias_factor_size)

        factors[:, self.target_attr] = target_factor_val
        factors[:, self.bias_attr] = bias_factor_val

        img = self.dataset.sample_observations_from_factors(factors, random_state=self.random_state)[0]
        img = (img * 255).astype(np.uint8)
        if img.shape[2] == 1:
            img = img[:, :, 0]
            img = Image.fromarray(img, 'L')
        else:
            img = Image.fromarray(img)
        img = self.transform(img)
        return img, torch.tensor([target_attr_binary_val], dtype=torch.float)
