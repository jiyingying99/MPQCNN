from matplotlib import pyplot as plt
from scipy.io import loadmat, savemat
import numpy as np
import os
import csv
import pandas as pd
from sklearn import preprocessing  # 0-1编码
from sklearn.model_selection import StratifiedShuffleSplit  # 随机划分，保证每一类比例相同

'''
Splitting the data set and generating noisy data
'''

# 根据输入信号和信噪比 (SNR) 生成高斯白噪声 (WGN)
def wgn(x, snr):
    snr = 10 ** (snr / 10.0)        # 将SNR从dB转换为线性标度
    xpower = np.sum(np.absolute(x) ** 2, axis=0) / x.shape[0]       # 计算平均信号功率
    npower = xpower / snr       # 根据信噪比计算噪声功率
    return np.random.standard_normal(x.shape) * np.sqrt(npower)     # 生成并返回具有计算功率的高斯白噪声


# 根据指定的 SNR 将高斯白噪声 (WGN) 添加到输入数据中
def add_noise(data, snr_num):
    rand_data = wgn(data, snr_num)      # 生成具有指定SNR的高斯白噪声
    n_data = data + rand_data       # 将生成的噪声添加到输入数据
    return n_data, rand_data        # 返回有噪声的数据和生成的噪声

def preprocess(d_path, noise=False, snr=0):

    # 获得该文件夹下所有.mat文件名
    filenames = os.listdir(d_path)
    print(filenames)

    # cwru capture函数从提供的文件中读取并捕获数据
    def capture():
        files = {}      # 初始化一个空字典以存储捕获的数据
        for i in filenames:     # 遍历文件名列表
            file_path = os.path.join(d_path, i)     # 获取完整的文件路径
            file = loadmat(file_path)       # 加载.mat文件
            file_keys = file.keys()     # 获取加载文件中的密钥
            print(file_keys)
            for key in file_keys:
                if 'DE' in key:     # 检查是否包含“DE”
                    files[i] = file[key].ravel()        # 将展开的数据存储在字典中
        print(files)
        return files        # 返回存储在字典中的捕获数据

    # 江南大学数据集
    # def capture():
    #     files = {}      # 初始化一个空字典以存储捕获的数据
    #     for i in filenames:     # 遍历文件名列表
    #         file_path = os.path.join(d_path, i)     # 获取完整的文件路径
    #         file = loadmat(file_path)       # 加载.mat文件
    #         file_keys = file.keys()     # 获取加载文件中的密钥
    #         print(file_keys)
    #         print(i)
    #         for key in file_keys:
    #             if 'data' in key:     # 检查是否包含“DE”
    #                 files[i] = file[key].ravel()        # 将展开的数据存储在字典中
    #     print(files)
    #     return files        # 返回存储在字典中的捕获数据

    # 东南大学数据集
    # def capture():
    #     files = {}      # 初始化一个空字典以存储捕获的数据
    #     for i in filenames:     # 遍历文件名列表
    #         file_path = os.path.join(d_path, i)     # 获取完整的文件路径
    #         file = loadmat(file_path)       # 加载.mat文件
    #         file_keys = file.keys()     # 获取加载文件中的密钥
    #         print(file_keys)
    #         print(i)
    #         for key in file_keys:
    #             if 'dndata' in key:     # 检查是否包含“DE”
    #                 files[i] = file[key].ravel()        # 将展开的数据存储在字典中
    #     print(files)
    #     return files        # 返回存储在字典中的捕获数据

    # 渥太华
    # def capture():
    #     files = {}      # 初始化一个空字典以存储捕获的数据
    #     for i in filenames:     # 遍历文件名列表
    #         file_path = os.path.join(d_path, i)     # 获取完整的文件路径
    #         file = loadmat(file_path)       # 加载.mat文件
    #         file_keys = file.keys()     # 获取加载文件中的密钥
    #         print(file_keys)
    #         print(i)
    #         for key in file_keys:
    #             if 'Channel_1' in key:     # 检查是否包含“DE”
    #                 files[i] = file[key].ravel()        # 将展开的数据存储在字典中
    #     print(files)
    #     return files        # 返回存储在字典中的捕获数据





    # 江南大学数据集
    # def capture():
    #     files = {}      # 初始化一个空字典以存储捕获的数据
    #     file_dict = {}
    #     s=[]
    #     for i in filenames:     # 遍历文件名列表
    #         file_path = os.path.join(d_path, i)     # 获取完整的文件路径
    #         with open(file_path, 'r') as file:
    #             reader = csv.reader(file)
    #             # headers = next(reader)  # 获取csv文件中的列名
    #             # content = headers.index(0)  # 提取csv文件中的指定列名
    #             for row in reader:
    #                 # s.append(row)
    #                 print(row)
    #
    #         # name, ext = os.path.splitext(file_path)
    #         # file_dict[name] = ext
    #         # file = loadmat(file_path)       # 加载.mat文件
    #     # files['ib600_2.mat'] = file['data'].ravel()  # 将展开的数据存储在字典
    #         # file_keys = file.keys()     # 获取加载文件中的密钥
    #         # for key in file_keys:
    #         #     if 'data' in key:     # 检查是否包含“DE”
    #         #         files[i] = file[key].ravel()  # 将展开的数据存储在字典中
    #     print(file_dict)
    #     return files        # 返回存储在字典中的捕获数据




    # 对提供的数据进行切片并向切片部分添加噪声
    # 该函数获取数据并将其分割为训练和测试部分。如果启用噪声，则会向训练和测试数据添加噪声。
    # 最后，它返回包含噪声数据和噪声的字典，用于训练和测试。
    # def slice_enc(data, noise, snr):
    #
    #     noised_data_dict1 = {}      # 初始化一个空字典以存储噪声数据以进行训练
    #     noise_dict1 = {}        # 初始化一个空字典以存储用于训练的噪声数据
    #     noised_data_dict2 = {}      # 初始化一个空字典以存储噪声数据以进行测试
    #     noise_dict2 = {}        # 初始化一个空字典以存储噪声数据以进行测试
    #
    #     for key, val in data.items():       # 遍历提供的数据字典中的项
    #         slice_data = val        # 获取切片数据
    #         # 前 2/3 的数据分配给train_data，剩余1/3的数据分配给test_data。
    #         train_data= slice_data[:len(slice_data) * 2 // 3]       # 对数据进行切片以进行训练
    #         test_data = slice_data[len(slice_data) * 2 // 3:]       # 对数据进行切片以进行测试
    #         if noise:       # 检查是否启用了噪音
    #             noised_data1, white_noise1 = add_noise(train_data, snr)     # 向训练数据添加噪声
    #             noised_data2, white_noise2 = add_noise(test_data, snr)      # 向测试数据添加噪声
    #
    #         noised_data_dict1[key] = noised_data1       # 将 噪声训练数据 存储在字典中
    #         noise_dict1[key] = white_noise1     # 将用于训练的 噪声数据 存储在字典中
    #         noised_data_dict2[key] = noised_data2       # 将 噪声测试数据 存储在字典中
    #         noise_dict2[key] = white_noise2     # 将用于测试的 噪声数据 存储在字典中
    #     return noised_data_dict1, noise_dict1, noised_data_dict2, noise_dict2       # 返回包含数据和噪声的词典

    def slice_enc(data):

        # noised_data_dict1 = {}      # 初始化一个空字典以存储噪声数据以进行训练
        # noise_dict1 = {}        # 初始化一个空字典以存储用于训练的噪声数据
        # noised_data_dict2 = {}      # 初始化一个空字典以存储噪声数据以进行测试
        # noise_dict2 = {}        # 初始化一个空字典以存储噪声数据以进行测试
        traindata = {}
        testdata = {}

        for key, val in data.items():       # 遍历提供的数据字典中的项
            slice_data = val        # 获取切片数据
            # 前 2/3 的数据分配给train_data，剩余1/3的数据分配给test_data。
            train_data= slice_data[:len(slice_data) * 2 // 3]       # 对数据进行切片以进行训练
            test_data = slice_data[len(slice_data) * 2 // 3:]       # 对数据进行切片以进行测试
            # if noise:       # 检查是否启用了噪音
            #     noised_data1, white_noise1 = add_noise(train_data, snr)     # 向训练数据添加噪声
            #     noised_data2, white_noise2 = add_noise(test_data, snr)      # 向测试数据添加噪声

            # noised_data_dict1[key] = noised_data1       # 将 噪声训练数据 存储在字典中
            # noise_dict1[key] = white_noise1     # 将用于训练的 噪声数据 存储在字典中
            # noised_data_dict2[key] = noised_data2       # 将 噪声测试数据 存储在字典中
            # noise_dict2[key] = white_noise2     # 将用于测试的 噪声数据 存储在字典中
            traindata[key] = train_data
            testdata[key] = test_data

        return traindata, testdata
        # return noised_data_dict1, noise_dict1, noised_data_dict2, noise_dict2       # 返回包含数据和噪声的词典

    # 将带有键“DE”的数据保存到位于以下位置的 .mat 文件中：
    def save_mat(d_path, train_data, test_data):
        path1 = d_path + '_TrainNoisedA'   # 定义保存噪声训练数据的路径
        if not os.path.exists(path1):       # 检查路径是否存在；如果没有，创建目录
            os.mkdir(path1)
        # 遍历字典中包含噪声测试数据的项目
        for k, dat in train_data.items():
            savemat(os.path.join(path1, k), {'DE': dat})        # 将数据保存在指定目录下的.mat文件中

        path1 = d_path + '_TestNoisedA'      # 定义保存噪声测试数据的路径
        if not os.path.exists(path1):
            os.mkdir(path1)
        # 遍历字典中包含噪声测试数据的项目
        for k, dat in test_data.items():
            savemat(os.path.join(path1, k), {'DE': dat})        # 将数据保存在指定目录下的.mat文件中

    # def save_mat(d_path, train_data, test_data):
    #     path1 = d_path + '_TrainNoised'  # 定义保存噪声训练数据的路径
    #     if not os.path.exists(path1):  # 检查路径是否存在；如果没有，创建目录
    #         os.mkdir(path1)
    #
    #     # 遍历字典中包含噪声训练数据的项目
    #     for k, dat in train_data.items():
    #         # 将数据从行转换为列
    #         dat_reshaped = np.reshape(dat, (len(dat), 1))
    #         # 将转换后的数据保存在指定目录下的.mat文件中
    #         savemat(os.path.join(path1, k), {'DE': dat_reshaped})
    #
    #     path1 = d_path + '_TestNoised'  # 定义保存噪声测试数据的路径
    #     if not os.path.exists(path1):
    #         os.mkdir(path1)
    #
    #     # 遍历字典中包含噪声测试数据的项目
    #     for k, dat in test_data.items():
    #         # 将数据从行转换为列
    #         dat_reshaped = np.reshape(dat, (len(dat), 1))
    #         # 将转换后的数据保存在指定目录下的.mat文件中
    #         savemat(os.path.join(path1, k), {'DE': dat_reshaped})

    # 江南大学数据集
    # def save_mat(d_path, train_data, test_data):
    #     path1 = d_path + '_TrainNoised_'
    #     if not os.path.exists(path1):
    #         os.mkdir(path1)
    #     for k, dat in train_data.items():
    #         savemat(os.path.join(path1, k), {'data': dat})
    #
    #     path1 = d_path + '_TestNoised_'
    #     if not os.path.exists(path1):
    #         os.mkdir(path1)
    #     for k, dat in test_data.items():
    #         savemat(os.path.join(path1, k), {'data': dat})

    # def save_mat(d_path, train_data, test_data):
    #     path1 = d_path + '_TrainNoised'  # 定义保存噪声训练数据的路径
    #     if not os.path.exists(path1):  # 检查路径是否存在；如果没有，创建目录
    #         os.mkdir(path1)
    #
    #     # 遍历字典中包含噪声训练数据的项目
    #     for k, dat in train_data.items():
    #         # 将数据从行转换为列
    #         dat_reshaped = np.reshape(dat, (len(dat), 1))
    #         # 将转换后的数据保存在指定目录下的.mat文件中
    #         savemat(os.path.join(path1, k), {'data': dat_reshaped})
    #
    #     path1 = d_path + '_TestNoised'  # 定义保存噪声测试数据的路径
    #     if not os.path.exists(path1):
    #         os.mkdir(path1)
    #
    #     # 遍历字典中包含噪声测试数据的项目
    #     for k, dat in test_data.items():
    #         # 将数据从行转换为列
    #         dat_reshaped = np.reshape(dat, (len(dat), 1))
    #         # 将转换后的数据保存在指定目录下的.mat文件中
    #         savemat(os.path.join(path1, k), {'data': dat_reshaped})


    # 东南大学数据集
    # def save_mat(d_path, train_data, test_data):
    #     path1 = d_path + '_TrainNoised'
    #     if not os.path.exists(path1):
    #         os.mkdir(path1)
    #     for k, dat in train_data.items():
    #         dat_reshaped = np.reshape(dat, (len(dat), 1))
    #         savemat(os.path.join(path1, k), {'dndata': dat_reshaped})
    #
    #     path1 = d_path + '_TestNoised_'
    #     if not os.path.exists(path1):
    #         os.mkdir(path1)
    #     for k, dat in test_data.items():
    #         dat_reshaped = np.reshape(dat, (len(dat), 1))
    #         savemat(os.path.join(path1, k), {'dndata': dat_reshaped})

    # def save_mat(d_path, train_data, test_data):
    #     path1 = d_path + '_TrainNoised'  # 定义保存噪声训练数据的路径
    #     if not os.path.exists(path1):  # 检查路径是否存在；如果没有，创建目录
    #         os.mkdir(path1)
    #
    #     # 遍历字典中包含噪声训练数据的项目
    #     for k, dat in train_data.items():
    #         # 将数据从行转换为列
    #         dat_reshaped = np.reshape(dat, (len(dat), 1))
    #         # 将转换后的数据保存在指定目录下的.mat文件中
    #         savemat(os.path.join(path1, k), {'dndata': dat_reshaped})
    #
    #     path1 = d_path + '_TestNoised'  # 定义保存噪声测试数据的路径
    #     if not os.path.exists(path1):
    #         os.mkdir(path1)
    #
    #     # 遍历字典中包含噪声测试数据的项目
    #     for k, dat in test_data.items():
    #         # 将数据从行转换为列
    #         dat_reshaped = np.reshape(dat, (len(dat), 1))
    #         # 将转换后的数据保存在指定目录下的.mat文件中
    #         savemat(os.path.join(path1, k), {'dndata': dat_reshaped})

    # 从所有.mat文件中读取出数据的字典
    data = capture()  #numpy.ndarray
    print(data)
    # 得到噪声数据和噪声
    # noised_data_dict1, noise_dict1, noised_data_dict2, noise_dict2 = slice_enc(data, noise, snr)
    train_data, test_data = slice_enc(data)
    # 保存得到的训练噪声数据和测试噪声数据
    save_mat(d_path, train_data, test_data)


if __name__ == "__main__":
    path = '../data/0HP'  # change dataset file folders to split data
    preprocess(d_path=path,
               noise=True
               )
