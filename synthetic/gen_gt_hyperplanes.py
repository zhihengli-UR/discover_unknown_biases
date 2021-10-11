import argparse
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from common.data_loader import preprocess_label, SHAPE_ATTRIBUTE_INDEX_PER_DATASET_DICT

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main(args):
    device = torch.device(0)

    pkl_fpath = os.path.join(args.data_root, '{}_{}.pkl'.format(args.real_alg_name, args.dset_name))
    with open(pkl_fpath, 'rb') as f:
        latent_code_dict = pickle.load(f)

    # compute normal vector for shape
    mu = latent_code_dict['mu']
    mu = torch.from_numpy(mu).to(device).float()

    attribute_labels = latent_code_dict['label']
    if args.dset_name == 'dsprites_full':
        attribute_labels = attribute_labels[:, 1:]
        attribute_labels[:, 0] = attribute_labels[:, 0] == 1  # convert 3 shape classes to binary classes (0, 1, 2) to (0, 1, 0)
    else:
        attribute_labels = preprocess_label(args.dset_name, attribute_labels)
        shape_attribute_index = SHAPE_ATTRIBUTE_INDEX_PER_DATASET_DICT[args.dset_name][0]
        reg_attribute_indices = torch.tensor([i for i in range(attribute_labels.shape[1])
                                                if i != shape_attribute_index], device=device)

    label = torch.from_numpy(attribute_labels).to(device).float()

    linear_layer = nn.Linear(mu.shape[1], attribute_labels.shape[1], bias=True).to(device)

    optimizer = torch.optim.Adam(params=linear_layer.parameters(), lr=args.lr)

    dataloader = DataLoader(TensorDataset(mu, label), batch_size=args.batch_size, shuffle=True,
                            num_workers=0)
    cls_criterion = nn.BCEWithLogitsLoss()
    mse_criterion = nn.MSELoss()

    smallest_loss = 1e10
    best_normal_vector_mat = None
    best_offset_mat = None

    tbar = tqdm(range(args.epoch))
    for e in tbar:
        total_loss = 0
        for idx, (data, label) in enumerate(dataloader):
            q = torch.linalg.qr(linear_layer.weight.t())[0].t()
            out = F.linear(data, q, linear_layer.bias)
            optimizer.zero_grad()

            if args.dset_name == 'dsprites_full':
                cls_loss = cls_criterion(out[:, 0], label[:, 0])
                mse_loss = mse_criterion(out[:, 1:], label[:, 1:])
            else:
                cls_loss = cls_criterion(out[:, shape_attribute_index], label[:, shape_attribute_index])
                mse_loss = mse_criterion(out[:, reg_attribute_indices], label[:, reg_attribute_indices])

            reg = sum([torch.norm(p, p=2) for p in [q, linear_layer.bias]])  # L2 regularization
            loss = cls_loss + mse_loss + reg

            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_loss = total_loss / len(dataloader)
        tbar.set_description('[{}/{}] loss: {:.3f}'.format(e, args.epoch, avg_loss))
        if avg_loss < smallest_loss:
            smallest_loss = avg_loss
            best_normal_vector_mat = linear_layer.weight.t().cpu().detach().numpy()  # no orthogonalization when saving normal vector matrix
            best_offset_mat = linear_layer.bias.cpu().detach().numpy()

    output_fpath = os.path.join(args.output_dir, '{}_{}.npz'.format(args.real_alg_name, args.dset_name))
    np.savez_compressed(output_fpath, normal_vector_mat=best_normal_vector_mat, offset_mat=best_offset_mat)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--dset_name', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=100000)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--num_workers', type=int, default=20)
    parser.add_argument('--real_alg_name', type=str, required=True)

    args = parser.parse_args()
    main(args)
