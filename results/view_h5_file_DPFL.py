import h5py
import matplotlib.pyplot as plt

# 打开HDF5文件


# file_path = 'mnist_FedPRF_001_1_1_0.h5'  # 替换为你的H5文件路径
file_path = 'mnist_DPFL_test_0.h5'

with h5py.File(file_path, 'r') as hf:
    # 读取数据集
    rs_test_acc_data = hf['rs_test_acc'][:]
    rs_train_loss_data = hf['rs_train_loss'][:]
    rs_server_acc_data = hf['rs_server_acc'][:]
    rs_server_loss_data = hf['rs_server_loss'][:]
    extra_msg = hf['extra_msg'][()]

    extra_msg = extra_msg.decode('utf-8')
    print("----------------------------------------------")
    print(extra_msg)
    print("rs_server_acc:", rs_server_acc_data)
    print("best_server_acc", max(rs_server_acc_data))
    print("rs_server_loss:", rs_server_loss_data)
    print("----------------------------------------------")

    # 打印数据
    # print("rs_test_acc:")
    # print(rs_test_acc_data)
    # print("rs_train_loss:")
    # print(rs_train_loss_data)

    print("================================================================")

    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(rs_test_acc_data, label='rs_test_acc')
    plt.plot(rs_train_loss_data, label='rs_train_loss')
    plt.plot(rs_server_acc_data, label='rs_server_acc')
    plt.plot(rs_server_loss_data, label='rs_server_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title(file_path)
    plt.legend()
    plt.show()
