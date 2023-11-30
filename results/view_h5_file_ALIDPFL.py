import h5py
import matplotlib.pyplot as plt

# 打开HDF5文件

# 替换为你的H5文件路径
file_path = 'mnist_ALIDPFL_002_01_4_0.h5'

with h5py.File(file_path, 'r') as hf:
    # 读取数据集
    rs_test_acc_data = hf['rs_test_acc'][:]
    rs_train_loss_data = hf['rs_train_loss'][:]
    rs_server_acc_data = hf['rs_server_acc'][:]
    rs_server_loss_data = hf['rs_server_loss'][:]
    rs_tau_list = hf['rs_tau_list'][:]
    extra_msg = hf['extra_msg'][()]

    extra_msg = extra_msg.decode('utf-8')
    print("----------------------------------------------")
    print(extra_msg)
    print("rs_server_acc:", rs_server_acc_data)
    print("best_server_acc", max(rs_server_acc_data))
    print("rs_server_loss:", rs_server_loss_data)
    print("rs_tau_list:", rs_tau_list)
    print("----------------------------------------------")

    # 打印数据
    # print("rs_test_acc:")
    # print(rs_test_acc_data)
    # print("rs_train_loss:")
    # print(rs_train_loss_data)

    print("================================================================")

    # 创建子图
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # 子图1：rs_test_acc_data和rs_server_acc_data
    axs[0].plot(rs_test_acc_data, label='rs_test_acc')
    axs[0].plot(rs_server_acc_data, label='rs_server_acc')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Value')
    axs[0].set_title(file_path)
    axs[0].legend()

    # 子图2：rs_train_loss_data和rs_server_loss_data
    axs[1].plot(rs_train_loss_data, label='rs_train_loss')
    axs[1].plot(rs_server_loss_data, label='rs_server_loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Value')
    axs[1].set_title(file_path)
    axs[1].legend()

    # 子图3：rs_tau_list
    axs[2].plot(rs_tau_list, label='rs_tau_list')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Value')
    axs[2].set_title(file_path)
    axs[2].legend()

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示图形
    plt.show()
