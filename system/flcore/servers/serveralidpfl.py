"""
Author: Xinpeng Ling
Target: Coding the server class of FedPRF algorithm
Email:  xpling@stu.ecnu.edu.cn
Home page: https://space.bilibili.com/3461572290677609
"""
import copy
import math
import os
import time
from random import random

import h5py
import numpy as np
import torch
import torch.nn as nn
import ujson

from flcore.clients.clientalidpfl import clientALIDPFL
from flcore.optimizers.utils.RDP.get_max_steps import get_max_steps
from flcore.servers.serverbase import Server
from threading import Thread

from utils.data_utils import read_server_testset
from system.flcore.optimizers.utils.RDP.compute_dp_sgd import apply_dp_sgd_analysis


def compute_new_tau(mu, C, Gamma, sigma, d, hat_B, Rs, Rc, tau_star):
    '''
    mu: 强凸系数
    C: Clipping norm
    Gamma: 体现异质性的系数
    sigma: 噪声乘子
    hat_B: 跟batchsize有关的一个值
    T = min{Rs·tau_star,Rc}
    '''
    T = min(Rs * tau_star, Rc)
    print(f"mu={mu}, C={C}, Gamma={Gamma}, sigma={sigma}, d={d}, "
          f"hat_B={hat_B}, Rs={Rs}, Rc={Rc}, T={T}, tau_star={tau_star}")
    dp_noise_bound = (sigma ** 2 * C ** 2 * d) / (hat_B ** 2)
    # 分子
    molecule = (4 / (mu ** 2)) + 3 * (C ** 2) + 2 * Gamma * T * mu + dp_noise_bound
    # 分母
    denominator = (2 + 2 / T) * (C ** 2 + dp_noise_bound)
    ret = math.sqrt(1 + molecule / (denominator + 1e-6))
    print(f"分子 = {molecule}, 分母 = {denominator}, 原始tau = {ret}")
    ret = int(ret + 0.5)  # 十分位四舍五入
    ret = max(1, ret)
    ret = min(ret, 100)
    return ret


def compute_l2_norm_of_model(model):
    l2_norm = 0.0
    model_dict = {}
    for key, var in model.state_dict().items():
        model_dict[key] = var.clone()

    for key in model_dict:
        l2_norm += model_dict[key].norm(2) ** 2

    l2_norm = l2_norm ** 0.5
    return l2_norm


def sub_model(model_1, model_2):
    ret_model = copy.deepcopy(model_1)
    for param in ret_model.parameters():  # 全局模型置为0，方便后面累加了，上一行深拷贝只是为了model shape
        param.data.zero_()
    for ret_model_param, model_1_param, model_2_param in zip(ret_model.parameters(), model_1.parameters(),
                                                             model_2.parameters()):
        a = model_1_param.data.clone()
        b = model_2_param.data.clone()
        ret_model_param.data = a - b
    return ret_model


def compute_mu(grad_l2_lists, model_l2_lists, weights):
    mu = 0.0
    for grad_l2, model_l2, w in zip(grad_l2_lists, model_l2_lists, weights):
        mu += w * grad_l2 / (model_l2 + 1e-6)
    return mu


class ALIDPFL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientALIDPFL)
        self.rs_server_acc = []  # 中心方测出来的准确率
        self.rs_server_loss = []  # 中心方测出来的loss，不是各client的loss的加权
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失函数,用来测server_loss的
        self.batch_sample_ratio = args.batch_sample_ratio
        self.dp_sigma = args.dp_sigma  # 算epsilon的时候要用
        self.dp_norm = args.dp_norm  # 裁剪范数C
        self.need_adaptive_tau = args.need_adaptive_tau
        self.tau_star = args.local_iterations  # optimal tau
        self.rs_tau_list = [self.tau_star]  # adap local iteration list
        self.dp_epsilon = args.dp_epsilon

        self.global_rounds = args.global_rounds
        self.local_iterations = args.local_iterations

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

        # $\hat{B}$用以下方式来估计
        self.hat_B = int(self.batch_sample_ratio * min([client.train_samples for client in self.clients]))
        self.dimension_of_model = sum(p.numel() for p in self.global_model.parameters())

        delta = 10 ** (-5)
        orders = (list(range(2, 64)) + [128, 256, 512])  # 默认的lamda
        self.Rc = get_max_steps(self.dp_epsilon, delta, self.batch_sample_ratio, self.dp_sigma, orders)
        print(f"===================== Rc={self.Rc} =====================")

        if self.need_adaptive_tau and self.global_rounds >= self.Rc:  # Rs>=Rc 取1最好
            self.need_adaptive_tau = False
            self.local_iterations = 1
            self.tau_star = 1
            for client in self.clients:
                client.need_adaptive_tau = False
                self.local_iterations = 1
                self.tau_star = 1

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
        # ALIDPFL的epsilon是预先设定好的隐私预算，用来算Rc的
        # 下面这种算法，是算epsilon预先不知道的
        # orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64)) + [128, 256, 512]
        # eps, opt_order = apply_dp_sgd_analysis(q=self.batch_sample_ratio,
        #                                        sigma=self.dp_sigma,
        #                                        # steps: 单个客户端本地迭代总轮数
        #                                        steps=sum(self.rs_tau_list),
        #                                        orders=orders,
        #                                        delta=10e-5)
        # print("eps:", format(eps) + "| order:", format(opt_order))

        if len(self.rs_test_acc):
            algo = algo + "_" + self.goal + "_" + str(self.times)  # goal的作用在这呢
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            extra_msg = f"dataset = {self.dataset}, learning_rate = {self.learning_rate},\n" \
                        f"rounds = {self.global_rounds}, batch_sample_ratio = {self.batch_sample_ratio},\n" \
                        f"num_clients = {self.num_clients}, algorithm = {self.algorithm} \n" \
                        f"have_PD = {self.args.privacy}, dp_sigma = {self.args.dp_sigma}\n" \
                        f"epsilon = {self.dp_epsilon}, dp_norm = {self.dp_norm}\n" \
                        f"Rs = {self.global_rounds}, Rc = {self.Rc}" \
                        f"need_adaptive_tau = {self.need_adaptive_tau}"
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
                hf.create_dataset('rs_tau_list', data=self.rs_tau_list[:-1])  # 最后一个τ并没有用上
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

    def send_models(self):  # sever->client
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_tau(self.tau_star)  # 发模型的时候，把tau_star发下去
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)  # 不太懂，为啥要乘2,上传+下发？

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)
        if self.need_adaptive_tau:
            self.global_model.eval()
            for client in self.clients:
                client.model.eval()

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():  # 全局模型置为0，方便后面累加了，上一行深拷贝只是为了model shape
            param.data.zero_()

        # print("本次聚合，各客户端权重为:", ["{:.4f}".format(weight) for weight in self.uploaded_weights])

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

        # 这里全局模型和本地模型都有，我就在这算mu和Gamma 并且算τ*
        if self.need_adaptive_tau:
            model_diff_l2_list = []
            grad_diff_l2_list = []
            loss_diff_list = []
            for client in self.clients:
                model_diff = sub_model(client.model, self.global_model)
                model_diff_l2_list.append(compute_l2_norm_of_model(model_diff))
                grad_diff = sub_model(client.grad_client_model, client.grad_global_model)
                grad_diff_l2_list.append(compute_l2_norm_of_model(grad_diff))
                loss_diff_list.append(client.loss_client_model - client.loss_global_model)

            self.mu_strong_convex = compute_mu(grad_diff_l2_list, model_diff_l2_list, self.uploaded_weights)
            # 敛析里的Gamma就是用绝对值来放大的，这样取没啥问题
            self.Gamma = abs(sum([w * loss_diff for w, loss_diff in zip(self.uploaded_weights, loss_diff_list)]))

            self.tau_star = compute_new_tau(
                mu=self.mu_strong_convex,
                C=self.dp_norm,
                Gamma=self.Gamma,
                sigma=self.dp_sigma,
                d=self.dimension_of_model,
                hat_B=self.hat_B,
                Rs=self.global_rounds,
                Rc=self.Rc,
                tau_star=self.tau_star
            )

            print(f"~~~~~~~~~ The τ of next round = {self.tau_star} ~~~~~~~~~")

        if self.need_adaptive_tau:
            self.rs_tau_list.append(self.tau_star)
        else:
            self.rs_tau_list.append(self.local_iterations)

    def train(self):
        for i in range(0, self.global_rounds + 1):
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
            print("adap_tau_list: ", self.rs_tau_list[:-1])
            if sum(self.rs_tau_list) > self.Rc:
                print(f"Rc={self.Rc} is running out")
                break
        if sum(self.rs_tau_list) <= self.Rc:
            print(f"Rs={self.global_rounds} is running out")

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
            self.set_new_clients(clientALIDPFL)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
