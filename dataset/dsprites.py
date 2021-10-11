import os
import numpy as np
import torch

from random import getrandbits
from PIL import Image
from torch.utils.data import Dataset


class dSprites(Dataset):
    def __init__(self, dset_dir, transform, target_attr: int, bias_attr: int, bias_degree=0.1):
        # attributes:
        # Color: white
        # Shape: square, ellipse, heart
        # Scale: 6 values linearly spaced in [0.5, 1]
        # Orientation: 40 values in [0, 2 pi]
        # Position X: 32 values in [0, 1]
        # Position Y: 32 values in [0, 1]

        self.transform = transform
        root = os.path.join(
            dset_dir, 'dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        npz = np.load(root)
        self.images = npz['imgs']
        self.labels = torch.tensor(npz['latents_classes'], dtype=torch.long)

        assert bias_attr >= 1
        assert target_attr >= 1
        assert target_attr != bias_attr

        self.target_attr = target_attr
        target_attr_mid_val = self.labels[:, target_attr].max().item() / 2
        # make target attribute to binary
        self.labels[:, target_attr] = (
            self.labels[:, target_attr] < target_attr_mid_val).long()
        self.bias_attr = bias_attr
        self.bias_attr_val_prob_conditioned_on_target_val = [
            [bias_degree, 1 - bias_degree], [1 - bias_degree, bias_degree]]
        len_biased_attr = len(self.labels[:, bias_attr].unique())
        # make biased attribute to binary
        self.labels[:, bias_attr] = (
            self.labels[:, bias_attr] < len_biased_attr / 2).long()

    def __getitem__(self, _):
        # randomly sample target attribtue value
        target_attr_val = getrandbits(1)
        biasd_attr_binary_val = np.random.choice(
            [0, 1], p=self.bias_attr_val_prob_conditioned_on_target_val[target_attr_val])
        indices = ((self.labels[:, self.target_attr] == target_attr_val) * (
            self.labels[:, self.bias_attr] == biasd_attr_binary_val)).nonzero(as_tuple=False)[:, 0]
        index = indices[np.random.randint(low=0, high=len(indices))]

        img = Image.fromarray(self.images[index] * 255)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[index, self.target_attr:self.target_attr + 1].float()

    def __len__(self):
        return len(self.images)

    def num_channels(self):
        return 1
