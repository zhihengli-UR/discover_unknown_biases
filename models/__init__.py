from .ae import AE
from .vae import VAE
from .betavae import BetaVAE
from .cvae import CVAE
from .ifcvae import IFCVAE


# TODO: add author and license info to all files.
# TODO: 3 different divergences in the InfoVAE paper https://arxiv.org/pdf/1706.02262.pdf
# TODO: evaluation metrics
# TODO: Add Adversarial Autoencoders https://arxiv.org/pdf/1511.05644.pdf
# TODO: A version of CVAE where independence between C and Z is enforced
# TODO: Add PixelCNN and PixelCNN++ and PixelVAE
# TODO: Add VQ-VAE (discrete encodings) and VQ-VAE2 --> I guess Version 2 has pixelCNN
# TODO: SCGAN_Disentangled_Representation_Learning_by_Addi

def set_args_by_model(model_name, args):
    model_to_args = {
        'VAE': {
            'alg': 'VAE',
            'encoder': 'SimpleGaussianConv64',
            'decoder': 'SimpleConv64'
        },
        'BetaVAE': {
            'alg': 'BetaVAE',
            'encoder': 'SimpleGaussianConv64',
            'decoder': 'SimpleConv64'
        },
        'BetaTCVAE': {
            'alg': 'BetaVAE',
            'encoder': 'PadlessGaussianConv64',
            'decoder': 'SimpleConv64',
            'controlled_capacity_increase': True,
            'loss_terms': 'BetaTCVAE',
            'discriminator': 'SimpleDiscriminator',
            'w_tc': 2.0,
            'lr_scheduler': 'ReduceLROnPlateau',
            'lr_scheduler_args': {'mode': 'min', 'factor': 0.8, 'patience': 0, 'min_lr': 0.000001},
            'iterations_c': 2000
        },
        'FactorVAE': {
            'alg': 'BetaVAE',
            'encoder': 'SimpleGaussianConv64',
            'decoder': 'SimpleConv64',
            'loss_terms': 'FactorVAE',
            'controlled_capacity_increase': True,
            'discriminator': 'SimpleDiscriminator',
            'w_tc': 1.0,
            'lr_scheduler': 'ReduceLROnPlateau',
            'lr_scheduler_args': {'mode': 'min', 'factor': 0.8, 'patience': 0, 'min_lr': 0.000001},
        },
        'DIPVAEI': {
            'alg': 'BetaVAE',
            'encoder': 'SimpleGaussianConv64',
            'decoder': 'SimpleConv64',
            'loss_terms': 'DIPVAEI',
            'discriminator': 'SimpleDiscriminator'
        },
        'DIPVAEII': {
            'alg': 'BetaVAE',
            'encoder': 'SimpleGaussianConv64',
            'decoder': 'SimpleConv64',
            'loss_terms': 'DIPVAEII',
            'discriminator': 'SimpleDiscriminator'
        }
    }

    model_args = model_to_args[model_name]
    for k, v in model_args.items():
        if k == 'encoder' or k == 'decoder' or k == 'loss_terms' or k == 'discriminator':
            v = [v]
        setattr(args, k, v)


def set_args_by_dataset(args):
    num_channels_per_dataset = {'dsprites_full': 1, 'smallnorb': 1}
    args.num_channels = num_channels_per_dataset[args.dset_name]
