import os
import torch
import argparse
import sys

from torchvision import transforms
from dataset.dsprites import dSprites
from dataset.disentanglement_lib_datasets import BiasedDisentanglementLibDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from architectures.encoders.simple_conv64 import SimpleConv64
from common.utils import set_environment_variables


class Trainer:
    def __init__(self, args):
        self.args = args
        transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor()])
        if args.dset_name == 'dsprites_full':
            dataset = dSprites(args.dset_dir, transform, args.target_attr_index,
                               args.bias_attr_index, bias_degree=args.bias_degree)
        else:
            dataset = BiasedDisentanglementLibDataset(
                args.dset_name, transform, args.target_attr_index, args.bias_attr_index, args.bias_degree)
        self.loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                 shuffle=True, pin_memory=True)
        self.device = torch.device(0)
        self.model = SimpleConv64(latent_dim=1, num_channels=dataset.num_channels(
        ), image_size=args.image_size).to(self.device)
        self.bce_criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), args.lr)
        self.total_epoch = args.epoch

        self.best_acc = -1

    def train(self, epoch):
        total_pred = 0
        total_loss = 0
        total_true_pos = 0
        self.model.train()
        pbar = tqdm(self.loader)
        for idx, (img, label) in enumerate(pbar):
            img = img.to(self.device)
            label = label.to(self.device)
            logits = self.model(img)
            loss = self.bce_criterion(logits, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            avg_loss = total_loss / (idx + 1)

            pred = torch.sigmoid(logits) >= 0.5
            true_pos = (pred.long() == label).sum().item()
            total_true_pos += true_pos
            total_pred += len(pred)

            pbar.set_description('[{}/{}] loss: {:.4f}, acc: {:.4f}'.format(epoch,
                                                                            self.total_epoch,
                                                                            avg_loss,
                                                                            total_true_pos / total_pred))

        acc = total_true_pos / total_pred
        if acc > self.best_acc:
            ckpt_fpath = os.path.join(args.ckpt_dir, '{}_target_{}_bias_{}.pth'.format(
                args.dset_name, args.target_attr_index, args.bias_attr_index))
            state = {
                'model': self.model.state_dict(),
                'epoch': epoch,
                'optimizer': self.optimizer.state_dict()
            }
            torch.save(state, ckpt_fpath)
            if acc > 0.999:
                sys.exit(0)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bias_attr_index', type=int, required=True)
    parser.add_argument('--dset_name', required=True,
                        choices=['dsprites_full', 'smallnorb'])
    parser.add_argument('--target_attr_index', type=int, required=True)
    parser.add_argument('--image_size', default=64, type=int)
    parser.add_argument('--dset_dir', default='data', type=str)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--ckpt_dir', type=str,
                        default='ckpt/synthetic/classifier')
    parser.add_argument('--bias_degree', type=float, default=0.1)
    args = parser.parse_args()
    assert os.path.exists(args.ckpt_dir)
    return args


if __name__ == '__main__':
    args = get_args()
    set_environment_variables(args.dset_dir, args.dset_name)
    trainer = Trainer(args)
    for e in range(1, args.epoch + 1):
        trainer.train(e)
