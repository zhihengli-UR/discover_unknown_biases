import sys
import torch
import os
import pickle
import models
import numpy as np

from tqdm import tqdm
from common.utils import setup_logging, initialize_seeds, set_environment_variables
from common.arguments import get_args
from models import set_args_by_model

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main(_args):
    # load the model associated with args.alg
    model_cl = getattr(models, _args.alg)
    model = model_cl(_args)

    result_dict = {'mu': [], 'label': []}

    # load checkpoint
    model.load_checkpoint(_args.ckpt_load, load_iternum=False, load_optim=False)

    model.net_mode(train=False)
    for x_true, label in tqdm(model.data_loader, dynamic_ncols=True):
        x_true = x_true.to(model.device)
        mu = model.encode_deterministic(images=x_true).cpu().numpy()
        result_dict['mu'].append(mu)
        result_dict['label'].append(label.numpy())

    result_dict['mu'] = np.concatenate(result_dict['mu'], axis=0)
    result_dict['label'] = np.concatenate(result_dict['label'], axis=0)

    fpath = os.path.join(_args.latent_vec_dir, '{}_{}.pkl'.format(_args.real_alg_name, _args.dset_name))
    with open(fpath, 'wb') as f:
        pickle.dump(result_dict, f)
    print('saved to {}'.format(fpath))


if __name__ == "__main__":
    _args = get_args(sys.argv[1:])
    assert _args.real_alg_name is not None
    assert _args.latent_vec_dir is not None
    if not os.path.exists(_args.latent_vec_dir):
        os.mkdir(_args.latent_vec_dir)

    _args.use_pbar = False
    _args.save_ckpt = False
    _args.traverse_z = False
    set_args_by_model(_args.real_alg_name, _args)

    setup_logging(_args.verbose)
    initialize_seeds(_args.seed)

    # set the environment variables for dataset directory and name, and check if the root dataset directory exists.
    set_environment_variables(_args.dset_dir, _args.dset_name)
    assert os.path.exists(os.environ.get('DISENTANGLEMENT_LIB_DATA', '')), \
        'Root dataset directory does not exist at: \"{}\"'.format(_args.dset_dir)

    with torch.no_grad():
        main(_args)
