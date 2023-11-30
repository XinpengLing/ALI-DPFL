import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.privacy import *


class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

    def train(self):
        print(f"Clinet {self.id} is training……")
        # trainloader = self.load_train_data()
        trainloader = self.load_train_data_minibatch(iterations=1)
        # self.model.to(self.device)
        self.model.train()  # 在训练开始之前写上 model.trian() ，在测试时写上 model.eval()

        # differential privacy
        if self.privacy:
            self.model, self.optimizer, trainloader, privacy_engine = \
                initialize_dp(self.model, self.optimizer, trainloader, self.dp_sigma)

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):
            # print(f"查看memory_summary(),mark 2,step = {step}\n", torch.cuda.memory_summary())
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:  # 这是什么操作哈哈，还故意慢慢训练吗，还是涉及到多线程之类的技巧问题:模拟算力异构
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)  # 前向传播
                loss = self.loss(output, y)
                self.optimizer.zero_grad()  # 梯度缓存清零，以确保每个训练批次的梯度都是从头开始计算的
                loss.backward()  # 对损失值 `loss` 进行反向传播，计算模型参数的梯度
                # loss.backward有很多操作: 1.内部有逐样本求梯度 2.逐样本的梯度裁剪
                self.optimizer.step()  # 梯度下降
                # .step 1.梯度求和 2.加噪 3.取平均 4.梯度下降

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")

        print(f"Clinet {self.id} had trained.")
