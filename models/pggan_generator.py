# python3.7
"""Contains the generator class of ProgressiveGAN.

Basically, this class is derived from the `BaseGenerator` class defined in
`base_generator.py`.
"""

import os
import numpy as np

import torch

from . import model_settings
from .pggan_generator_model import PGGANGeneratorModel
from .base_generator import BaseGenerator

__all__ = ['PGGANGenerator']


class PGGANGenerator(BaseGenerator):
    """Defines the generator class of ProgressiveGAN."""

    def __init__(self, model_name, logger=None):
        super().__init__(model_name, logger)
        assert self.gan_type == 'pggan'

    def build(self):
        self.check_attr('fused_scale')
        self.model = PGGANGeneratorModel(resolution=self.resolution,
                                         fused_scale=self.fused_scale,
                                         output_channels=self.output_channels)

    def load(self):
        self.logger.info(f'Loading pytorch model from `{self.model_path}`.')
        self.model.load_state_dict(torch.load(self.model_path))
        self.logger.info(f'Successfully loaded!')
        self.lod = self.model.lod.to(self.cpu_device).tolist()
        self.logger.info(f'  `lod` of the loaded model is {self.lod}.')

    def sample(self, num):
        assert num > 0
        return np.random.randn(num, self.latent_space_dim).astype(np.float32)

    def preprocess(self, latent_codes):
        if not isinstance(latent_codes, np.ndarray):
            raise ValueError(
                f'Latent codes should be with type `numpy.ndarray`!')

        latent_codes = latent_codes.reshape(-1, self.latent_space_dim)
        norm = np.linalg.norm(latent_codes, axis=1, keepdims=True)
        latent_codes = latent_codes / norm * np.sqrt(self.latent_space_dim)
        return latent_codes.astype(np.float32)

    def synthesize(self, latent_codes):
        latent_codes_shape = latent_codes.shape
        if not (len(latent_codes_shape) == 2 and
                latent_codes_shape[1] == self.latent_space_dim):
            raise ValueError(f'Latent_codes should be with shape [batch_size, '
                             f'latent_space_dim], where `batch_size` no larger than '
                             f'{self.batch_size}, and `latent_space_dim` equal to '
                             f'{self.latent_space_dim}!\n'
                             f'But {latent_codes_shape} received!')

        images = self.model(latent_codes)
        return images
