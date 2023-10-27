import os
import torchvision
import torchvision.transforms as transforms
from datasets.utils import *
import global_v as glv


def get_cifar100(data_path, network_config):
    print("loading CIFAR100")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    trans_list = [transforms.RandomCrop(32, padding=4),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                  Cutout(length=16),
                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ]
    if glv.network_config['optimizer'] == 'SGD':
        del trans_list[3]
    print('Transform list is:', trans_list)
    
    transform_train = transforms.Compose(trans_list)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform_test)

    return trainset, testset
