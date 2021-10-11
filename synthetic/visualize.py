import sys
import os
import torch
import numpy as np
import models

from common.arguments import get_args
from common.utils import initialize_seeds
from common.linalg import project_point_to_plane, traverse_along_normal_vec
from torchvision.utils import save_image
from common.constants import SYNTHETIC_ATTRIBUTE_NAME
from common.utils import get_data_for_visualization


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(0)

        model_cl = getattr(models, args.alg)
        self.model = model_cl(args)
        self.model.load_checkpoint(args.ckpt_load)
        self.model.net_mode(train=False)
        print('generative model checkpoint loaded.')

        attribute_names = SYNTHETIC_ATTRIBUTE_NAME[args.dset_name]
        self.biased_attribute_name = attribute_names[args.bias_attr_index]
        self.target_attribute_name = attribute_names[args.target_attr_index]

        self.traverse_steps = torch.linspace(
            args.start_distance, args.end_distance, args.steps, device=self.device).unsqueeze(0).float().reshape(1, -1, 1)

        pred_bias_hyperplane = np.load(args.pred_bias_hyperplane_fpath)
        self.pred_bias_normal_vec = torch.from_numpy(
            pred_bias_hyperplane['normal_vector'][0]).to(self.device)
        self.pred_bias_offset = torch.from_numpy(
            pred_bias_hyperplane['offset'][0]).to(self.device)

    def visualize(self):
        sample_images_dict, sample_labels_dict = get_data_for_visualization(
            self.model.data_loader.dataset, self.device)
        encodings = dict()
        for key in sample_images_dict.keys():
            encodings[key] = self.model.encode_deterministic(
                images=sample_images_dict[key], labels=sample_labels_dict[key]).detach()

        for key in encodings:
            z = encodings[key]
            label_orig = sample_labels_dict[key]

            w = self.pred_bias_normal_vec
            b = self.pred_bias_offset
            img_lst = []

            w = w.unsqueeze(0)
            b = b.reshape(1, 1)
            z_projected = project_point_to_plane(z, w, b)
            z_traverse = traverse_along_normal_vec(
                z_projected, w, self.traverse_steps, None)

            out_image = self.model.decode(
                latent=z_traverse.reshape(-1, self.args.z_dim), labels=label_orig).cpu()
            out_image = out_image.reshape(self.traverse_steps.shape[1], args.num_channels,
                                          self.args.image_size, self.args.image_size)
            img_lst.append(out_image.cpu().numpy())

            imgs_all = torch.tensor(
                img_lst).reshape(-1, args.num_channels, self.args.image_size, self.args.image_size)
            fpath = os.path.join(args.vis_dir, '{}_b_{}_t_{}.jpg'.format(
                key, self.biased_attribute_name, self.target_attribute_name))
            save_image(imgs_all, fpath,
                       nrow=self.traverse_steps.shape[1], pad_value=1)


if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    initialize_seeds(args.seed)
    assert args.real_alg_name is not None
    models.set_args_by_model(args.real_alg_name, args)
    models.set_args_by_dataset(args)

    assert args.dset_name in SYNTHETIC_ATTRIBUTE_NAME
    assert args.bias_attr_index is not None
    assert args.target_attr_index is not None

    if args.dset_name == 'dsprites_full':
        args.bias_attr_index -= 1
        args.target_attr_index -= 1

    assert args.pred_bias_hyperplane_fpath is not None
    assert os.path.exists(args.pred_bias_hyperplane_fpath)

    if args.fname_suffix != '':
        args.fname_suffix = '_' + args.fname_suffix

    if not os.path.exists(args.vis_dir):
        os.mkdir(args.vis_dir)

    trainer = Trainer(args)
    with torch.no_grad():
        trainer.visualize()
