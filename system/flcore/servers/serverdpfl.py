"""
Author: Xinpeng Ling
Target: Coding the server class of FedPRF algorithm
Email:  xpling@stu.ecnu.edu.cn
Home page: https://space.bilibili.com/3461572290677609
"""
import os
import time
from random import random

import h5py
import numpy as np
import torch
import torch.nn as nn
import ujson

from flcore.clients.clientdpfl import clientDPFL
from flcore.servers.serverbase import Server
from threading import Thread

from utils.data_utils import read_server_testset
from system.flcore.optimizers.utils.RDP.compute_dp_sgd import apply_dp_sgd_analysis


def weighted_variance(data, weights):
    if len(data) != len(weights):
        raise ValueError("The lengths of data and weights must be the same.")

    weighted_mean = sum(data[i] * weights[i] for i in range(len(data))) / sum(weights)
    weighted_squared_diff = sum(weights[i] * (data[i] - weighted_mean) ** 2 for i in range(len(data)))
    weighted_variance = weighted_squared_diff / sum(weights)

    return weighted_variance


class DPFL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientDPFL)
        self.rs_server_acc = []  # 中心方测出来的准确率
        self.rs_server_loss = []  # 中心方测出来的loss，不是各client的loss的加权
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失函数,用来测server_loss的
        self.batch_sample_ratio = args.batch_sample_ratio
        self.dp_sigma = args.dp_sigma  # 算epsilon的时候要用

        self.global_rounds = args.global_rounds
        self.local_iterations = args.local_iterations

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"

        current_path = os.path.abspath(__file__)  # 获取当前脚本的绝对路径
        parent_directory = os.path.dirname(current_path)  # 找到当前脚本的父目录
        parent_directory = os.path.dirname(parent_directory)  # 找到父目录的父目录
        parent_directory = os.path.dirname(parent_directory)  # system
        root_directory = os.path.dirname(parent_directory)  # 项目根目录的绝对路径
        config_json_path = root_directory + "\\dataset\\" + self.dataset + "\\config.json"

        if not os.path.exists(result_path):
            os.makedirs(result_path)

        # 计算一下隐私 epsilon
        orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64)) + [128, 256, 512]
        eps, opt_order = apply_dp_sgd_analysis(q=self.batch_sample_ratio,
                                               sigma=self.dp_sigma,
                                               # steps: 单个客户端本地迭代总轮数
                                               steps=self.global_rounds * self.local_iterations,
                                               orders=orders,
                                               delta=10e-5)
        print("eps:", format(eps) + "| order:", format(opt_order))

        if len(self.rs_test_acc):
            algo = algo + "_" + self.goal + "_" + str(self.times)  # goal的作用在这呢
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            extra_msg = f"dataset = {self.dataset}, learning_rate = {self.learning_rate},\n" \
                        f"rounds = {self.global_rounds}, batch_sample_ratio = {self.batch_sample_ratio},\n" \
                        f"num_clinets = {self.num_clients}, algorithm = {self.algorithm} \n" \
                        f"have_PD = {self.args.privacy}, dp_sigma = {self.args.dp_sigma}\n" \
                        f"epsilon = {eps}\n"
            with open(config_json_path) as f:
                data = ujson.load(f)

            extra_msg = extra_msg + "--------------------config.json------------------------\n" \
                                    "num_clients={}, num_classes={}\n" \
                                    "non_iid={}, balance={},\n" \
                                    "partition={}, alpha={}\n".format(
                data["num_clients"], data["num_classes"], data["non_iid"],
                data["balance"], data["partition"], data["alpha"])

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.create_dataset('rs_server_acc', data=self.rs_server_acc)
                hf.create_dataset('rs_server_loss', data=self.rs_server_loss)
                hf.create_dataset('extra_msg', data=extra_msg, dtype=h5py.string_dtype(encoding='utf-8'))

    def evaluate_server(self, q=0.2, test_batch_size=64):
        """
        中心方做一下评估，拿一下acc和loss
        方式是，把client的测试集并到一起，得到一个server_testset
        用这个测试集进行评估
        """
        test_loader_full = read_server_testset(self.dataset, q=q, batch_size=test_batch_size)
        self.global_model.eval()  # 开启测试模式
        test_acc = 0
        test_num = 0
        with torch.no_grad():
            for x, y in test_loader_full:  # 一个batch 一个batch 来
                if type(x) == type([]):  # 这组判断是把数据加载到GPU/CPU里
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.global_model(x)
                loss = self.loss(output, y)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()  # 这里其实是一个计数器，不是准确率
                test_num += y.shape[0]
        accuracy = test_acc / test_num
        self.rs_server_acc.append(accuracy)
        self.rs_server_loss.append(loss.item())
        print("Accuracy at server: {:.4f}".format(accuracy))
        print("Loss at server: {:.4f}".format(loss))

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()  # 下发模型

            if i % self.eval_gap == 0:  # 几轮测试一次全局模型
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model by personalized")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()

            if self.dlg_eval and i % self.dlg_gap == 0:  # 算 峰值信噪比
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)  # 本轮的时间开销
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if i % self.eval_gap == 0:  # 几轮测试一次全局模型
                print("\nEvaluate global model by global")
                self.evaluate_server(q=0.2, test_batch_size=64)

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("Best local_avg_accuracy={:.4f}, Last local_avg_accuracy={:.4f}".format(
            max(self.rs_test_acc), self.rs_test_acc[-1]))
        print("Best server_accuracy={:.4f}, Last server_accuracy={:.4f}".format(
            max(self.rs_server_acc), self.rs_server_acc[-1]))
        print("Last server_loss={:.4f}".format(self.rs_server_loss[-1]))
        print("Average time cost per round={:.4f}".format(sum(self.Budget[1:]) / len(self.Budget[1:])))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientDPFL)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
