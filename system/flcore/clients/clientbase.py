import copy
import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data


# 该采样每次抽取的数量相等
class EquallySizedAndIndependentBatchSampler:
    def __init__(self, dataset, minibatch_size, iterations):
        self.length = len(dataset)
        self.minibatch_size = minibatch_size
        self.iterations = iterations

    def __iter__(self):
        for _ in range(self.iterations):
            # numpy.random.choice(a, size=None, replace=True, p=None)
            # 从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组
            # replace:True表示可以取相同数字（有放回），False表示不可以取相同数字（无放回）
            # 数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同
            yield np.random.choice(self.length, self.minibatch_size, replace=False)  # 这里最后产生的是一组坐标

    def __len__(self):
        return self.iterations


# 该采样每次采样的数量不等，即泊松采样，minibatch_size是采样多次后的平均采样数
class IIDBatchSampler:
    def __init__(self, dataset, minibatch_size, iterations):
        self.length = len(dataset)  # 总的样本个数
        self.minibatch_size = minibatch_size  # batch大小
        self.iterations = iterations  # batch个数

    def __iter__(self):
        for _ in range(self.iterations):  # 共iterations次

            # torch.rand：返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数，这里返回length个即总样本个数的随机数
            # self.minibatch_size / self.length=256/4500
            # np.where:输出满足条件 (即非0) 元素的坐标 (等价于numpy.nonzero)。这里的坐标以tuple的形式给出
            # 也就是说这里每轮会有不固定的坐标数，为什么每轮要不固定而不能固定呢？到底采样的意思是什么呢？这里的意思就是每轮的采样数量不等，但是最后的采样数量平均数等于给定的值
            indices = np.where(torch.rand(self.length) < (self.minibatch_size / self.length))[0]  # 这里具体得到坐标的数值
            # print("*************")
            # print(indices)
            if indices.size > 0:
                yield indices

    def __len__(self):
        return self.iterations


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset  # str
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs

        # check BatchNorm
        self.has_BatchNorm = False  # 这段代码用于检查神经网络模型中是否包含 批归一化（Batch Normalization）层
        for layer in self.model.children():  # 迭代神经网络模型的所有子层
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma

        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失函数
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        # 指数衰减的学习率调度器
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_train_data_minibatch(self, minibatch_size=None, iterations=None):
        if minibatch_size is None:
            minibatch_size = self.batch_size
        if iterations is None:
            iterations = 1
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        # 这个等于是不放回的按比例采样
        # return DataLoader(train_data,
        #                   batch_sampler=EquallySizedAndIndependentBatchSampler(train_data, minibatch_size, iterations))

        # # 这个是泊松采样
        return DataLoader(train_data,
                          batch_sampler=IIDBatchSampler(train_data, minibatch_size, iterations)
                          )

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)

    def set_parameters(self, model):  # 覆盖model.parameters()的操作；是get/set这种类型的操作
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()  # 深拷贝

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()  # 设置成“测试模式”,简单理解成不用反向传播了

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloaderfull:  # 一个batch 一个batch 来
                if type(x) == type([]):  # 这组判断是把数据加载到GPU/CPU里
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                # output的形状是[batch_size「=2」,num_classes「=3」],这里dim=1的意思是，找到num_classes里面输出概率最大的值，
                # 举例: output=[[0.15,0.22,0.98],[0.11,0.85,0.02]],那经过dim=1的筛选以后，就得到[2,1]
                # 通过指定 `dim=1`，`torch.argmax(output, dim=1)` 的操作将在每个样本的预测输出中寻找最大值所在的索引。
                # 换句话说，它会返回一个形状为 `[batch_size]` 的张量，其中的每个元素表示对应样本的预测输出中具有最大值的类别索引。
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()  # 这里其实是一个计数器，不是准确率
                test_num += y.shape[0]

                # AUC相关指标的计算
                # y_prob.append(output.detach().cpu().numpy())
                # nc = self.num_classes
                # if self.num_classes == 2:  # 这一步好像是二分类问题所需要的，具体原因没看明白
                #     nc += 1
                # lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                # if self.num_classes == 2:
                #     lb = lb[:, :2]
                # y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        # 下面的部分是用来计算AUC值的,是一种模型性能指标
        # 准确率和AUC是两个不同的指标，用于评估分类模型的性能，它们分别从整体准确性和类别排序能力的角度来衡量模型的表现。
        # y_prob = np.concatenate(y_prob, axis=0)
        # y_true = np.concatenate(y_true, axis=0)
        #
        # auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        auc = 0  # 这里不好搞，FedPRF会报错，反正我也用不上这个指标，就不算了

        return test_acc, test_num, auc

    def train_metrics(self):  # 这个train_metrics更像是用来对标test_metracs的，并不是模型训练 train_model，目的是拿到loss
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]  # .item是将张量转换成标量，转换完就不能当张量用了(求导、反向传播)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        # 这里loss是总损失，loss/train_num才是平均损失
        return losses, train_num

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y

    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))
