"""
Author: Xinpeng Ling
Target: Coding the client class of FedPRF algorithm
Email:  xpling@stu.ecnu.edu.cn
Home page: https://space.bilibili.com/3461572290677609
"""
import copy

import torch
import torch.nn as nn
import numpy as np
import time

from flcore.clients.clientbase import Client
from flcore.optimizers.dp_optimizer import DPAdam, DPSGD

from utils.privacy import *
import torch.nn.utils as utils


class clientDPFL(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.dp_norm = args.dp_norm
        self.batch_sample_ratio = args.batch_sample_ratio
        self.auto_s = args.auto_s
        self.local_iterations = args.local_iterations

        if self.privacy:
            self.optimizer = DPSGD(
                l2_norm_clip=self.dp_norm,  # 裁剪范数
                noise_multiplier=self.dp_sigma,
                minibatch_size=self.batch_size,  # batch_size
                microbatch_size=1,  # 几个样本梯度进行一次裁剪
                # 后面这些参数是继承父类的（SGD优化器的一些参数）
                params=self.model.parameters(),
                lr=self.learning_rate,
            )

    def train(self):
        print("---------------------------------------")
        print(f"Client {self.id} is training, privacy={self.privacy}, AUTO-S={self.auto_s}")
        minibatch_size = int(self.train_samples * self.batch_sample_ratio)
        trainloader = self.load_train_data_minibatch(minibatch_size=minibatch_size,
                                                     iterations=self.local_iterations)

        self.model.train()  # 在训练开始之前写上 model.trian() ，在测试时写上 model.eval()

        start_time = time.time()

        max_local_epochs = self.local_epochs  # 在DP里，一般epoch = 1，甚至都不会跑完全部的数据，只会来几个iterations
        if self.train_slow:  # False
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):  # FedPRF中限定epochs=1
            for i, (x, y) in enumerate(trainloader):  # load_train_data_minibatch，只有一个batch，循环只有一次
                print(f"Clint {self.id} 的第 {i+1} 次 iterations, 本次采样个数len(y): {len(y)}")
                self.optimizer.zero_accum_grad()  # 梯度清空
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                loss_batchs_list = []
                for j, (x_single, y_single) in enumerate(zip(x, y)):  # 遍历每个样本
                    self.optimizer.zero_microbatch_grad()  # 梯度清空
                    output = self.model(torch.unsqueeze(x_single.to(torch.float32), 0))  # 逐样本的原因，这里x要升维
                    loss = self.loss(output, torch.unsqueeze(y_single.to(torch.long), 0))  # 逐样本的原因，这里y要升维
                    loss_batchs_list.append(loss.item())  # 把逐样本的loss收集起来，后面打印
                    loss.backward()  # 求导得到梯度

                    self.optimizer.microbatch_step(self.privacy, self.auto_s)  # 这里做每个样本的梯度裁剪和梯度累加操作

                # 这里没法if self.privacy，因为优化器这我改不动
                self.optimizer.step_dp()  # 这里做的是梯度加噪和梯度平均更新下降的操作

                self.loss_batch_avg = sum(loss_batchs_list) / len(loss_batchs_list)  # 对逐样本的loss做平均，回传
                print("client {} is training, loss_batch_avg = {}".format(self.id, self.loss_batch_avg))

        if self.learning_rate_decay:  # False
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
