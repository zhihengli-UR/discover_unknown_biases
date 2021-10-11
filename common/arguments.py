import argparse
import os

from common import constants as c
from common.utils import str2bool, StoreDictKeyPair


def update_args(args):
    args.ckpt_load_iternum = False
    args.file_save = True
    args.gif_save = True
    return args


def get_args(sys_args):
    parser = argparse.ArgumentParser(description='disentanglement-pytorch')
    parser.add_argument('--evaluation_metric', default=None, type=str, choices=c.EVALUATION_METRICS, nargs='+',
                        help='Metric to evaluate the model during training')

    # name
    parser.add_argument('--alg', type=str, help='the disentanglement algorithm', choices=c.ALGS)
    parser.add_argument('--controlled_capacity_increase', help='to use controlled capacity increase', default=False)
    parser.add_argument('--loss_terms', help='loss terms to be incldued in the objective', nargs='*',
                        default=list(), choices=c.LOSS_TERMS)
    parser.add_argument('--name', default='unknown_experiment', type=str, help='name of the experiment')

    # Neural architectures
    parser.add_argument('--encoder', type=str, nargs='+', choices=c.ENCODERS,
                        help='name of the encoder network')
    parser.add_argument('--decoder', type=str, nargs='+', choices=c.DECODERS,
                        help='name of the decoder network')
    parser.add_argument('--label_tiler', type=str, nargs='*', choices=c.TILERS,
                        help='the tile network used to convert one hot labels to 2D channels')
    parser.add_argument('--discriminator', type=str, nargs='*', choices=c.DISCRIMINATORS,
                        help='the discriminator network')

    # Test or train
    parser.add_argument('--test', default=False, type=str2bool, help='to test')

    # training hyper-params
    parser.add_argument('--max_iter', default=3e5, type=lambda x: int(float(x)), help='maximum training iteration')
    parser.add_argument('--max_epoch', default=3e5, type=float, help='maximum training epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--num_disc_layers', default=5, type=int, help='number of fc layers in discriminators')
    parser.add_argument('--size_disc_layers', default=1000, type=int, help='size of fc layers in discriminators')

    # latent encoding
    parser.add_argument('--z_dim', default=10, type=int, help='size of the encoded z space')
    parser.add_argument('--include_labels', default=None, type=str, nargs='*',
                        help='Labels (indices or names) to include in latent encoding.')
    parser.add_argument('--l_dim', default=0, type=str, help='size of the encoded w space (for each label)')

    # optimizer
    parser.add_argument('--beta1', default=0.9, type=float, help='beta1 parameter of the Adam optimizer')
    parser.add_argument('--beta2', default=0.999, type=float, help='beta2 parameter of the Adam optimizer')
    parser.add_argument('--lr_G', default=1e-4, type=float, help='learning rate of the main autoencoder')
    parser.add_argument('--lr_D', default=1e-4, type=float, help='learning rate of all the discriminators')
    parser.add_argument('--D_beta1', default=0.5, type=float, help='beta1 for discriminator')
    parser.add_argument('--D_beta2', default=0.9, type=float, help='beta2 for discriminator')

    # Neural architectures hyper-parameters
    parser.add_argument('--num_layer_disc', default=6, type=int, help='number of fc layers in discriminators')
    parser.add_argument('--size_layer_disc', default=1000, type=int, help='size of fc layers in discriminators')

    # Loss weights and parameters [Common]
    parser.add_argument('--w_recon', default=1.0, type=float, help='reconstruction loss weight')
    parser.add_argument('--w_kld', default=1.0, type=float, help='main KLD loss weight (e.g. in BetaVAE)')

    # Loss weights and parameters for [CapacityVAE]
    parser.add_argument('--max_c', default=25.0, type=float, help='maximum value of control parameter in CapacityVAE')
    parser.add_argument('--iterations_c', default=100000, type=int, help='how many iterations to reach max_c')

    # Loss weights and parameters for [FactorVAE & BetaTCVAE]
    parser.add_argument('--w_tc', default=1.0, type=float,
                        help='total correlation loss weight (e.g. in FactorVAE and BetaTCVAE)')

    # Loss weights and parameters for [InfoVAE]
    parser.add_argument('--w_infovae', default=1.0, type=float,
                        help='mmd loss weight (e.g. in InfoVAE)')

    # Loss weights and parameters for [DIPVAE I & II]
    parser.add_argument('--w_dipvae', default=1.0, type=float,
                        help='covariance regularizer loss weight (e.g. in DIPVAE I and II)')

    # Loss weights and parameters for [IFCVAE]
    parser.add_argument('--w_le', default=1.0, type=float, help='label encoding loss weight (e.g. in IFCVAE)')
    parser.add_argument('--w_aux', default=1.0, type=float, help='auxiliary discriminator loss weight (e.g. in IFCVAE)')

    # Hyperparameters for [DIP-VAE]
    parser.add_argument('--lambda_d_factor', default=10.0, type=float,
                        help='Hyperparameter for diagonal values of covariance matrix')
    parser.add_argument('--lambda_od', default=1.0, type=float,
                        help='Hyperparameter for off diagonal values of covariance matrix.')

    # Dataset
    parser.add_argument('--dset_dir', default=os.getenv('DISENTANGLEMENT_LIB_DATA', './data'),
                        type=str, help='main dataset directory')
    parser.add_argument('--dset_name', default=None, type=str, help='dataset name')
    parser.add_argument('--image_size', default=64, type=int, help='width and height of image')
    parser.add_argument('--num_workers', default=0, type=int, help='number of workers for the data loader')
    parser.add_argument('--pin_memory', default=True, type=str2bool,
                        help='pin_memory flag of data loader. Check this blogpost for details:'
                             'https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/')

    # Logging and visualization
    parser.add_argument('--train_output_dir', default='train_outputs', type=str, help='output directory')
    parser.add_argument('--test_output_dir', default='test_outputs', type=str, help='test output directory')
    parser.add_argument('--file_save', default=True, type=str2bool, help='whether to save generated images to file')
    parser.add_argument('--gif_save', default=True, type=str2bool, help='whether to save generated GIFs to file')
    parser.add_argument('--traverse_spacing', default=0.2, type=float, help='spacing to traverse latents')
    parser.add_argument('--traverse_min', default=-2, type=float, help='min limit to traverse latents')
    parser.add_argument('--traverse_max', default=+2, type=float, help='max limit to traverse latents')
    parser.add_argument('--traverse_z', default=False, type=str2bool, help='whether to traverse the z space')
    parser.add_argument('--traverse_l', default=False, type=str2bool, help='whether to traverse the l space')
    parser.add_argument('--traverse_c', default=False, type=str2bool, help='whether to traverse the condition')
    parser.add_argument('--verbose', default=20, type=int, help='verbosity level')

    # Save/Load checkpoint
    parser.add_argument('--ckpt_dir', default='ckpt', type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_load', default=None, type=str, help='checkpoint name to load')
    parser.add_argument('--ckpt_load_iternum', default=True, type=str2bool, help='start global iteration from ckpt')
    parser.add_argument('--ckpt_load_optim', default=True, type=str2bool, help='load the optimizer state')
    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--use_pbar', action='store_true')

    # Iterations [default for all is equal to 1 epoch]
    parser.add_argument('--treat_iter_as_epoch', default=False, type=bool, help='treat all iter arguments as epochs')
    parser.add_argument('--ckpt_save_iter', default=None, type=lambda x: int(float(x)), help='iters to save checkpoint [default: 1 epoch]')
    parser.add_argument('--evaluate_iter', default=None, type=lambda x: int(float(x)), help='iters to evaluate [default: 1 epoch]')
    parser.add_argument('--float_iter', default=None, type=lambda x: int(float(x)), help='iters to aggregate float logs [default: 1 epoch]')
    parser.add_argument('--print_iter', default=None, type=lambda x: int(float(x)), help='iters to print float values [default: 1 epoch]')
    parser.add_argument('--all_iter', default=None, type=lambda x: int(float(x)), help='use same iteration for all [default: 1 epoch]')
    parser.add_argument('--recon_iter', default=None, type=lambda x: int(float(x)), help='iters to reconstruct image [default: 1 epoch]')
    parser.add_argument('--traverse_iter', default=None, type=lambda x: int(float(x)), help='iters to visualize latents [default: 1 epoch]')
    parser.add_argument('--schedulers_iter', default=None, type=lambda x: int(float(x)), help='iters to apply scheduler [default: 1 epoch]')

    # Schedulers
    parser.add_argument('--lr_scheduler', default=None, type=str, choices=c.LR_SCHEDULERS,
                        help='Type of learning rate scheduler [default: no scheduler]')
    parser.add_argument("--lr_scheduler_args", dest='lr_scheduler_args', action=StoreDictKeyPair,
                        nargs="+", metavar="KEY=VAL", help="Arguments of the for the lr_scheduler. See PyTorch docs.")
    parser.add_argument('--w_recon_scheduler', default=None, type=str, choices=c.SCHEDULERS,
                        help='Type of scheduler for the reconstruction weight [default: no scheduler]')
    parser.add_argument("--w_recon_scheduler_args", dest='w_recon_scheduler_args', action=StoreDictKeyPair,
                        nargs="+", metavar="KEY=VAL", help="Arguments of the for the w_recon_scheduler.")

    # Other
    parser.add_argument('--seed', default=123, type=int, help='Seed value for random, torch, cuda, and numpy.')

    # bias discovery
    parser.add_argument('--latent_vec_dir', type=str)
    parser.add_argument('--bias_attr_index', type=int)
    parser.add_argument('--target_attr_index', type=int)
    parser.add_argument('--biased_classifier_ckpt', type=str)
    parser.add_argument('--normal_vector_npz_fpath', type=str)
    parser.add_argument('--lambda_orthogonal_constraint', type=float, default=0.0)
    parser.add_argument('--lambda_hessian_penalty', type=float, default=0.0)
    parser.add_argument('--vis_root', type=str, default='vis')
    parser.add_argument('--save_traversal', action='store_true')
    parser.add_argument('--save_traversal_interval', type=int, default=100)
    parser.add_argument('--hessian_penalty', action='store_true')
    parser.add_argument('--init_axis', type=int)
    parser.add_argument('--real_alg_name', type=str)
    parser.add_argument('--biased_normal_vec_save_dir', type=str)
    parser.add_argument('--max_iter_biased_norm_vec', type=lambda x: int(float(x)), default=int(1e3))
    parser.add_argument('--optimizer', choices=['SGD', 'Adam'], default='Adam')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--pred_baseline_normal_vec_mat_fpath', type=str)
    parser.add_argument('--baseline_eval_results_dir', type=str)
    parser.add_argument('--bias_degree', type=float, default=0.1)
    parser.add_argument('--num_channels', type=int, default=1)
    parser.add_argument('--bias_gen', action='store_true')

    # visualize traversal
    parser.add_argument('--fname_suffix', type=str, default='')
    parser.add_argument('--pred_bias_hyperplane_fpath', type=str)
    parser.add_argument('--start_distance', type=float, default=-3.0)
    parser.add_argument('--end_distance', type=float, default=5.0)
    parser.add_argument('--steps', type=int, default=6)

    args = parser.parse_args(sys_args)

    args.num_labels = 0
    if args.include_labels is not None:
        args.num_labels = len(args.include_labels)

    # test
    args = update_args(args) if args.test else args

    # make sure arguments for supplementary neural architectures are included
    if c.FACTORVAE in args.loss_terms:
        assert args.discriminator is not None, 'The FactorVAE algorithm needs a discriminator to test the ' \
                                               'permuted latent factors ' \
                                               '(try the flag: --discriminator=SimpleDiscriminator)'

    args.vis_dir = os.path.join(args.vis_root, args.name)

    if args.bias_attr_index is not None and args.target_attr_index is not None:
        assert args.bias_attr_index != args.target_attr_index

    if args.bias_gen:
        assert args.dset_name != 'dsprites_full'

    return args
