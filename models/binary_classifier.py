import torch.nn as nn
from torchvision.models import vgg19_bn, resnet18


def get_binary_classifier(arch='vgg'):
    print(f'binary classifier architecture: {arch}.')
    if arch == 'vgg':
        model = vgg19_bn(pretrained=True)
        model.classifier[-1] = nn.Linear(4096, 1)
    elif arch == 'resnet':
        model = resnet18(pretrained=True)
        fc_dim = model.fc.weight.shape[1]
        model.fc = nn.Linear(fc_dim, 1)
    else:
        raise NotImplementedError

    return model
