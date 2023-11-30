import os
import ujson
import numpy as np
import gc
from sklearn.model_selection import train_test_split

batch_size = 10
train_size = 0.75  # merge original training set and test set, then split it manually.
least_samples = batch_size / (1 - train_size)  # least samples for each client
alpha = 1.0  # for Dirichlet distribution


def check(config_path, train_path, test_path, num_clients, num_classes, niid=False,
          balance=True, partition=None):
    # check existing dataset
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = ujson.load(f)
        if config['num_clients'] == num_clients and \
                config['num_classes'] == num_classes and \
                config['non_iid'] == niid and \
                config['balance'] == balance and \
                config['partition'] == partition and \
                config['alpha'] == alpha and \
                config['batch_size'] == batch_size:
            print("\nDataset already generated.\n")
            return True

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return False


def separate_data(data, num_clients, num_classes, niid=False, balance=False, partition=None, class_per_client=2):
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data  # 传进来的时候就是二元组；数据分为两个部分，一个是 原始内容，一个是 标签

    dataidx_map = {}

    if not niid:  # IID的情况
        partition = 'pat'
        class_per_client = num_classes

    if partition == 'pat':
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = []
        for i in range(num_classes):
            # 这个idxs[dataset_label==i]相当于一个循环了, 把所有dataset_label==i的下标，append到idx_for_each_class里面
            idx_for_each_class.append(idxs[dataset_label == i])

        class_num_per_client = [class_per_client for _ in range(num_clients)]  # 这样的话，每个client的种类都是一样的了，类似McMahan分片了
        for i in range(num_classes):  # 处理每一个类
            selected_clients = []
            for client in range(num_clients):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
                # 下面这个是 num_clients*(class_per_client/num_classes)
                # 等于说，采样率q = (class_per_client/num_classes)，然后按采样率去截取一些clients
                # 这里不太懂为啥采样率这样取，那不是变成了，每个client有2个种类，采样率就0.2；有3个种类，采样率就0.3；采样率跟种类有关？
                # 答：主要原因还是这里是IID，MNIST总共有10个种类，那每个Client都有10个种类，采样率就是1；
                selected_clients = selected_clients[:int(np.ceil((num_clients / num_classes) * class_per_client))]

            num_all_samples = len(idx_for_each_class[i])  # 拿到数字“i”的个数
            num_selected_clients = len(selected_clients)  # 拿到被采样客户端的数量
            num_per = num_all_samples / num_selected_clients  # 每个客户端能拿到多少个“i”
            if balance:  # 这边应该是代码复用了，IID只能balance=True，没有else分支
                num_samples = [int(num_per) for _ in range(num_selected_clients - 1)]
            else:
                num_samples = np.random.randint(max(num_per / 10, least_samples / num_classes), num_per,
                                                num_selected_clients - 1).tolist()
            num_samples.append(num_all_samples - sum(num_samples))
            # 把剩下的samples分给最后一个client，这里是应对上面的num_per取整的问题，不算严格的IID了，但是问题不大

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx + num_sample]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx + num_sample],
                                                    axis=0)
                    # 这行代码将 `idx_for_each_class[i][idx:idx + num_sample]` 数组中的行添加到 `dataidx_map[client]` 数组的末尾，使得 `dataidx_map[client]` 数组在第一个维度上增长。
                    # 通过指定 `axis=0`，表示沿着第一个维度（轴）进行拼接
                idx += num_sample
                class_num_per_client[client] -= 1

    elif partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        # 猎户座大佬：https://zhuanlan.zhihu.com/p/468992765
        min_size = 0
        K = num_classes
        N = len(dataset_label)

        while min_size < num_classes:  #
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]  # 获取类别k的图像数据的索引
                np.random.shuffle(idx_k)  # 打乱索引列表
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))  # 生成一个狄利克雷的比例列表
                # 下面这个bool式(len(idx_j) < N / num_clients)，控制idx_batch有没有达到batchsize，如果是False的话，p*0=0
                proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]
    else:
        raise NotImplementedError

    # assign data
    for client in range(num_clients):
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]

        for i in np.unique(y[client]):  # 从给定的数组中提取出唯一的元素；对提取出的唯一元素进行排序（默认情况下按升序排序）；返回一个包含唯一元素的新数组
            statistic[client].append((int(i), int(sum(y[client] == i))))  # 统计，每个client拿了多少个数据“i”

    del data
    # gc.collect()

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    # X: 记录了每个client拥有的数据的 原始内容 的 下标
    # y: 记录了每个client拥有的数据的   标签   的 下标
    # statistic: 记录了每个client拥有的数据类型及数量
    return X, y, statistic


def split_data(X, y):
    # Split dataset
    train_data, test_data = [], []
    num_samples = {'train': [], 'test': []}

    for i in range(len(y)):  # i是client的意思
        X_train, X_test, y_train, y_test = train_test_split(
            X[i], y[i], train_size=train_size, shuffle=True)

        train_data.append({'x': X_train, 'y': y_train})
        num_samples['train'].append(len(y_train))
        test_data.append({'x': X_test, 'y': y_test})
        num_samples['test'].append(len(y_test))

    print("Total number of samples:", sum(num_samples['train'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    print("The number of test samples:", num_samples['test'])
    print()
    del X, y
    # gc.collect()

    return train_data, test_data


def save_file(config_path, train_path, test_path, train_data, test_data, num_clients,
              num_classes, statistic, niid=False, balance=True, partition=None):
    config = {
        'num_clients': num_clients,
        'num_classes': num_classes,
        'non_iid': niid,
        'balance': balance,
        'partition': partition,
        'Size of samples for labels in clients': statistic,
        'alpha': alpha,
        'batch_size': batch_size,
    }

    # gc.collect()
    print("Saving to disk.\n")

    for idx, train_dict in enumerate(train_data):
        with open(train_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_dict)
    for idx, test_dict in enumerate(test_data):
        with open(test_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_dict)
    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("Finish generating dataset.\n")
