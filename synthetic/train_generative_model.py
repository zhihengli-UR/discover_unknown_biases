import sys
import torch
import os

from common.utils import setup_logging, initialize_seeds, set_environment_variables
from common.arguments import get_args
import models
from models import set_args_by_model

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main(_args):
    # load the model associated with args.alg
    model_cl = getattr(models, _args.alg)
    model = model_cl(_args)

    # load checkpoint
    if _args.ckpt_load:
        model.load_checkpoint(_args.ckpt_load, load_iternum=_args.ckpt_load_iternum, load_optim=_args.ckpt_load_optim)

    # run test or train
    if not _args.test:
        model.train()
    else:
        model.test()


if __name__ == "__main__":
    _args = get_args(sys.argv[1:])
    _args.use_pbar = True
    _args.save_ckpt = True
    _args.traverse_z = True
    set_args_by_model(_args.real_alg_name, _args)
    setup_logging(_args.verbose)
    initialize_seeds(_args.seed)

    if _args.bias_gen:
        assert _args.target_attr_index is not None
        assert _args.bias_attr_index is not None
        assert _args.target_attr_index != _args.bias_attr_index

    # set the environment variables for dataset directory and name, and check if the root dataset directory exists.
    set_environment_variables(_args.dset_dir, _args.dset_name)
    assert os.path.exists(os.environ.get('DISENTANGLEMENT_LIB_DATA', '')), \
        'Root dataset directory does not exist at: \"{}\"'.format(_args.dset_dir)

    main(_args)
