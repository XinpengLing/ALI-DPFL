import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms

from generate_server_testset import generate_server_testset
from utils.dataset_utils import check, separate_data, split_data, save_file

random.seed(1)
np.random.seed(1)
num_clients = 10
num_classes = 10
dir_path = "mnist/"


# Allocate data to users
def generate_mnist(dir_path, num_clients, num_classes, niid, balance, partition, need_server_testset=False):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    # 这里是做一个检查，如果config.json文件已经存在，并且里面的参数与这次要生成的文件，参数都一样的话，就直接return
    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
        return

    # FIX HTTP Error 403: Forbidden
    from six.moves import urllib
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    # Get MNIST data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    # 当 `download=True` 时，如果数据集尚未下载到指定路径中，函数会自动从官方源下载数据集并保存到指定路径中。下载完成后，数据集将被加载到内存中供后续使用。
    trainset = torchvision.datasets.MNIST(
        root=dir_path + "rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(
        root=dir_path + "rawdata", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())  # 看这里，他把trainset和testset的数据拢在一起了
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)  # 拢在一起的测试集+数据集，叫做dataset
    dataset_label = np.array(dataset_label)

    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    # X: 记录了每个client拥有的数据的 原始内容 的 下标
    # y: 记录了每个client拥有的数据的   标签   的 下标
    # statistic: 记录了每个client拥有的数据类型及数量
    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,
                                    niid, balance, partition)
    train_data, test_data = split_data(X, y)

    if need_server_testset:
        generate_server_testset(test_data, test_path)

    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
              statistic, niid, balance, partition)


if __name__ == "__main__":
    # niid = True if sys.argv[1] == "noniid" else False
    # balance = True if sys.argv[2] == "balance" else False
    # partition = sys.argv[3] if sys.argv[3] != "-" else None
    # need_server_testset = True if sys.argv[4] == "FL" else False

    niid = True
    balance = False
    partition = 'dir'
    need_server_testset = True

    generate_mnist(dir_path, num_clients, num_classes, niid, balance, partition, need_server_testset)
