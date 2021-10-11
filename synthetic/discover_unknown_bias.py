import sys
import os
import torch
import numpy as np
import models
import torch.nn.functional as F

from tqdm import tqdm
from common.arguments import get_args
from architectures.encoders.simple_conv64 import SimpleConv64
from common.utils import initialize_seeds
from common.linalg import project_point_to_plane, traverse_along_normal_vec


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(0)

        self.biased_classifier = SimpleConv64(
            latent_dim=1, num_channels=args.num_channels, image_size=args.image_size).to(self.device)
        self.biased_classifier.load_state_dict(
            torch.load(args.biased_classifier_ckpt)['model'])
        print('biased classifier checkpoint loaded.')
        self.biased_classifier.eval()

        model_cl = getattr(models, args.alg)
        self.model = model_cl(args)
        self.model.load_checkpoint(args.ckpt_load)
        self.model.net_mode(train=False)
        print('generative model checkpoint loaded.')

        w = torch.randn(1, args.z_dim, device=self.device)
        b = torch.randn(1, 1, device=self.device)

        self.w = torch.nn.Parameter(w, requires_grad=True)
        self.b = torch.nn.Parameter(b, requires_grad=True)
        if args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD([self.w, self.b], args.lr)
        elif args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(
                [self.w, self.b], args.lr, betas=(args.beta1, args.beta2))
        else:
            raise NotImplementedError

        hyperplanes = np.load(args.normal_vector_npz_fpath)
        gt_normal_vector_mat = torch.from_numpy(
            hyperplanes['normal_vector_mat']).float()
        gt_orth_normal_vector_mat = torch.linalg.qr(gt_normal_vector_mat)[0]
        assert gt_orth_normal_vector_mat.shape == gt_normal_vector_mat.shape
        self.gt_biased_normal_vec = gt_orth_normal_vector_mat[:, args.bias_attr_index].to(
            self.device).reshape(1, -1)
        self.gt_target_normal_vec = gt_orth_normal_vector_mat[:, args.target_attr_index].to(
            self.device).reshape(1, -1)
        gt_offset_mat = torch.from_numpy(hyperplanes['offset_mat'])
        self.gt_biased_offset = gt_offset_mat[args.bias_attr_index].to(
            self.device)

        gt_other_normal_vectors = gt_normal_vector_mat[:, [i for i in range(
            gt_normal_vector_mat.shape[1]) if i != args.bias_attr_index]].to(self.device)
        gt_orth_other_normal_vectors = torch.qr(gt_other_normal_vectors)[0]
        assert gt_orth_other_normal_vectors.shape == gt_other_normal_vectors.shape

        self.unit_gt_other_normal_vectors = gt_orth_other_normal_vectors / \
            torch.norm(gt_orth_other_normal_vectors, dim=0).unsqueeze(0)

        self.traverse_steps = torch.arange(
            args.traverse_min, args.traverse_max, args.traverse_spacing, device=self.device).reshape(1, -1, 1)

    def train(self):
        total_loss = 0
        total_tv_loss = 0
        total_orthogonal_constraint_loss = 0

        pbar = tqdm(range(args.max_iter_biased_norm_vec), dynamic_ncols=True)
        for idx in pbar:
            z = torch.rand(self.args.batch_size,
                           self.args.z_dim, device=self.device)

            z_projected = project_point_to_plane(z, self.w, self.b)
            z_traverse = traverse_along_normal_vec(
                z_projected, self.w, self.traverse_steps)

            out = self.model.decode(
                latent=z_traverse.reshape(-1, self.args.z_dim))
            prob = torch.sigmoid(self.biased_classifier(out)).reshape(
                self.args.batch_size, self.traverse_steps.shape[1])
            # total variation loss
            tv_loss = - \
                torch.log(1e-10 + torch.abs(prob[:, 1:] - prob[:, :-1]).mean())
            total_tv_loss += tv_loss.item()
            orthogonal_constraint_loss = (
                ((self.w / torch.norm(self.w)) @ self.unit_gt_other_normal_vectors) ** 2).mean()
            total_orthogonal_constraint_loss += orthogonal_constraint_loss.item()
            loss = tv_loss + self.args.lambda_orthogonal_constraint * orthogonal_constraint_loss

            self.optimizer.zero_grad()
            self.biased_classifier.zero_grad()
            self.model.model.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            avg_loss = total_loss / (idx + 1)
            with torch.no_grad():
                cosine_sim = F.cosine_similarity(
                    self.w, self.gt_biased_normal_vec)
                target_cosine_sim = F.cosine_similarity(
                    self.w, self.gt_target_normal_vec)

            desc = 'loss: {:.3f}, normal_vec_cos: {:.3f}, ' \
                   'target_vec_cos_sim: {:.3f}'.format(avg_loss, cosine_sim.item(),
                                                       target_cosine_sim.item())
            pbar.set_description(desc)


if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    initialize_seeds(args.seed)
    args.use_pbar = False
    args.save_ckpt = False
    args.traverse_z = False
    assert args.real_alg_name is not None
    models.set_args_by_model(args.real_alg_name, args)
    models.set_args_by_dataset(args)
    assert args.biased_normal_vec_save_dir is not None
    assert os.path.exists(args.biased_normal_vec_save_dir)
    assert args.bias_attr_index is not None
    assert args.target_attr_index is not None
    if args.dset_name == 'dsprites_full':
        args.bias_attr_index -= 1
        args.target_attr_index -= 1
    assert os.path.exists(args.biased_classifier_ckpt)
    assert args.normal_vector_npz_fpath is not None
    assert os.path.exists(args.normal_vector_npz_fpath)
    assert os.path.exists(args.ckpt_load)

    trainer = Trainer(args)
    trainer.train()

    fpath = os.path.join(args.biased_normal_vec_save_dir,
                         '{}.npz'.format(args.name))
    np.savez_compressed(fpath, normal_vector=trainer.w.cpu(
    ).detach().numpy(), offset=trainer.b.cpu().detach().numpy())
    print('optimized w and b are saved to {}'.format(fpath))
