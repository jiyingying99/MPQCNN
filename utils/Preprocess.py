import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import os
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit

'''
Preprocess for training and test
Refer to https://github.com/AaronCosmos/wdcnn_bearning_fault_diagnosis
'''


# def prepro(d_path, length=2048, number=1000, normal=True, enc=True, enc_step=28, snr=0, property='Train', noise=True):
#     """对数据进行预处理,返回train_X, train_Y, valid_X, valid_Y, test_X, test_Y样本.
#
#         :param d_path: 源数据地址
#         :param length: 信号长度，默认2个信号周期，864
#         :param number: 每种信号个数,总共10类,默认每个类别1000个数据
#         :param normal: 是否标准化.True,Fales.默认True
#         :param enc: 训练集、验证集是否采用数据增强.Bool,默认True
#         :param enc_step: 增强数据集采样顺延间隔
#
#         ```
#         import preprocess.preprocess_nonoise as pre
#
#         ```
#         """
#
#     # if (property == 'Train') & (noise == True):     # 当想要处理训练噪声数据时
#     #     d_path = d_path +'_TrainNoised_' + str(snr)
#     # elif (property == 'Test') & (noise == True):        # 当想要处理测试噪声数据时
#     #     d_path = d_path +'_TestNoised_' + str(snr)
#     # elif noise ==False:     # 当想要处理原始无噪声数据时
#     #     d_path = d_path
#
#     if property == 'Train':     # 当想要处理训练噪声数据时
#         d_path = d_path +'_TrainNoised'
#     elif property == 'Test':     # 当想要处理测试噪声数据时
#         d_path = d_path +'_TestNoised'
#
#     # 获得该文件夹下所有.mat文件名
#     filenames = os.listdir(d_path)
#
#     def capture_mat():
#         files = {}  # 初始化一个空字典以存储捕获的数据
#         for i in filenames:  # 遍历文件名列表
#             file_path = os.path.join(d_path, i)  # 获取完整的文件路径
#             file = loadmat(file_path)  # 加载.mat文件
#             file_keys = file.keys()  # 获取加载文件中的密钥
#             for key in file_keys:
#                 if 'DE' in key:  # 检查是否包含“DE”
#                     files[i] = file[key].ravel()  # 将展开的数据存储在字典中
#         return files  # 返回存储在字典中的捕获数据
#
#     # 江南大学数据集
#     # def capture_mat():
#     #     files = {}
#     #     for i in filenames:
#     #         file_path = os.path.join(d_path, i)
#     #         file = loadmat(file_path)
#     #         file_keys = file.keys()
#     #         for key in file_keys:
#     #             if 'data' in key:
#     #                 files[i] = file[key].ravel()
#     #     return files
#
#     # 东南大学数据集
#     # def capture_mat():
#     #     files = {}
#     #     for i in filenames:
#     #         file_path = os.path.join(d_path, i)
#     #         file = loadmat(file_path)
#     #         file_keys = file.keys()
#     #         for key in file_keys:
#     #             if 'dndata' in key:
#     #                 files[i] = file[key].ravel()
#     #     # print(files)
#     #     return files
#
#     # 渥太华数据集
#     # def capture_mat():
#     #     files = {}
#     #     for i in filenames:
#     #         file_path = os.path.join(d_path, i)
#     #         file = loadmat(file_path)
#     #         file_keys = file.keys()
#     #         for key in file_keys:
#     #             if 'Channel_1' in key:
#     #                 files[i] = file[key].ravel()
#     #     return files
#
#     def slice_enc(data):
#         keys = data.keys()      # 从字典中检索所有键data并将它们分配给keys
#         Train_Samples = {}      # 初始化一个空字典Train_Samples来存储生成的训练样本
#         for i in keys:
#             slice_data = data[i]        # 对于每个键i，检索与该键对应的数据并将其分配给变量slice_data
#             all_length = len(slice_data)        # 计算slice_data的总长度并将其分配给变量all_length
#             end_index = int(all_length)     # 将all_length转换为整数并将其分配给变量end_index
#             samp_train = int(number)        # 将number的值转换为整数并将其分配给变量samp_train。该变量表示要生成的训练样本的数量。
#             Train_sample = []       # 初始化Train_sample，存储当前数据集生成的训练样本
#
#             # 数据增强
#             if enc:
#                 enc_time = length // enc_step       # 计算样本长度为length可以符合一步大小为enc_step的数据的个数
#                 samp_step = 0       # 初始化一个变量samp_step以跟踪生成的样本数量
#                 for j in range(samp_train):
#                     random_start = np.random.randint(low=0, high=(end_index - 2 * length))      # 在一个范围内随机选择一个起点，以确保length可以从数据中生成至少两个大小的样本
#                     label = 0       # 初始化一个标签0来指示尚未达到所需的样本数量
#                     for h in range(enc_time):
#                         samp_step += 1
#                         random_start += enc_step        # 根据更新值以移动到下一个位置enc_step
#                         sample = slice_data[random_start: random_start + length]        # 从当前位置的数据中提取大小的样本
#                         Train_sample.append(sample)
#                         if samp_step == samp_train:
#                             label = 1       # label为1，表示已经生成了所需数量的样本
#                             break
#                     if label:
#                         break
#
#             else:
#                 for j in range(samp_train):  # 抓取训练数据
#                     random_start = np.random.randint(low=0, high=(end_index - length))
#                     sample = slice_data[random_start:random_start + length]
#                     Train_sample.append(sample)
#             Train_Samples[i] = Train_sample
#
#         return Train_Samples
#
#
#
#
#     # 将输入的数据集拆分为特征向量列表X和标签列表Y，其中X包含所有样本的特征向量，Y包含对应的标签
#     def add_labels(train_test):
#         X = []
#         Y = []
#         label = 0
#         for i in filenames:
#             x = train_test[i]       # 获取数据
#             X += x      # 将x追加到X列表中
#             lenx = len(x)       # 获取x的长度len(x)
#             Y += [label] * lenx     # 创建一个由重复元素label组成的列表，长度为lenx，并将其追加到Y列表中
#             label += 1      # 通过递增label的值来为不同类别的数据点赋予不同的标签
#         return X, Y
#
#     # 将输入的标签集进行独热编码处理
#     # 将每个标签编码为一个只包含0和1的向量，其中每个标签对应的位置为1，其余位置为0
#     def one_hot(Train_Y):
#         Train_Y = np.array(Train_Y).reshape([-1, 1])        # 将Train_Y转换为NumPy数组，并使用reshape函数将其重新整形为二维数组，其中第一维为-1，第二维为1
#         # 实例化了一个名为Encoder的OneHotEncoder对象，并将其与训练标签数据Train_Y进行拟合
#         # 在调用fit函数时，OneHotEncoder会根据输入的Train_Y数据学习如何对每个唯一的类别进行独热编码转换
#         # 编码器会了解到数据集中存在多少个不同的类别，并且可以为每个类别分配一个唯一的独热编码
#         Encoder = preprocessing.OneHotEncoder()
#         Encoder.fit(Train_Y)
#         # 使用Encoder对Train_Y进行转换，并将其转换为数组形式
#         Train_Y = Encoder.transform(Train_Y).toarray()
#         # 将编码后的标签转换回原始的标签
#         Train_Y = Encoder.inverse_transform(Train_Y)
#         # 将Train_Y转换为NumPy数组
#         Train_Y = np.asarray(Train_Y, dtype=float)
#         return Train_Y
#
#     # 归一化[-1, 1]
#     def scalar_stand(Train_X):
#         scalar = preprocessing.StandardScaler().fit(Train_X)        # 该fit方法计算Train_X中每个特征的平均值和标准差
#         Train_X = scalar.transform(Train_X)     # 标准化数据。transform方法使用拟合过程中获得的平均值和标准差来缩放数据。这种标准化可确保特征的平均值为0，标准差为1。
#         return Train_X
#
#     # 划分测试集
#     # 约 1/3 的数据被用作验证集，而剩余的 2/3 数据被用作训练集
#     def valid_test_slice(Test_X, Test_Y):
#         test_size = 3 / 17
#         # StratifiedShuffleSplit 是一个交叉验证迭代器，它生成用于划分数据集的索引
#         # 将数据集随机拆分成训练集和测试集，并确保训练集和测试集中的类别分布相同
#         # n_splits 参数设置为 1，表示仅生成一组划分
#         ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
#         for train_index, test_index in ss.split(Test_X, Test_Y):
#             X_valid, X_test = Test_X[train_index], Test_X[test_index]
#             Y_valid, Y_test = Test_Y[train_index], Test_Y[test_index]
#             # np.random.seed(200)
#             # np.random.shuffle(X_valid)
#             # np.random.seed(200)
#             # np.random.shuffle(Y_valid)
#             #
#             # np.random.seed(200)
#             # np.random.shuffle(X_test)
#             # np.random.seed(200)
#             # np.random.shuffle(Y_test)
#             return X_valid, Y_valid, X_test, Y_test
#
#     def wgn(x, snr):
#         snr = 10 ** (snr / 10.0)  # 将SNR从dB转换为线性标度
#         xpower = np.sum(np.absolute(x) ** 2, axis=0) / x.shape[0]  # 计算平均信号功率
#         npower = xpower / snr  # 根据信噪比计算噪声功率
#         return np.random.standard_normal(x.shape) * np.sqrt(npower)  # 生成并返回具有计算功率的高斯白噪声
#
#     # 根据指定的 SNR 将高斯白噪声 (WGN) 添加到输入数据中
#     def add_noise(data, snr_num):
#         rand_data = wgn(data, snr_num)  # 生成具有指定SNR的高斯白噪声
#         n_data = data + rand_data  # 将生成的噪声添加到输入数据
#         return n_data  # 返回有噪声的数据和生成的噪声
#
#     # 从所有.mat文件中读取出数据的字典
#     data = capture_mat()
#     # 获取样本数据
#     train = slice_enc(data)
#     # 制作标签，返回X，Y
#     Train_X, Train_Y = add_labels(train)
#     # One-hot标签
#     Train_Y = one_hot(Train_Y)
#     # 需要做一个数据转换，转换成np格式
#     Train_X = np.asarray(Train_X)
#     plt.figure(1)
#     plt.plot(Train_X[28])
#     plt.show()
#     if noise == True:
#         Train_X = add_noise(Train_X, snr)
#         plt.figure(2)
#         plt.plot(Train_X[28])
#         plt.show()
#     # Train_Y = np.asarray(Train_Y)
#
#     # 是否标准化
#     if normal:
#         Train_X = scalar_stand(Train_X)
#
#     if property == 'Train':
#         # 将训练集切分为训练集和验证集
#         Train_X1, Train_Y, Valid_X1, Valid_Y = valid_test_slice(Train_X, Train_Y)
#         return Train_X1, Train_Y, Valid_X1, Valid_Y
#
#     if property == 'Test':
#         return Train_X, Train_Y
#
#
# if __name__ == "__main__":
#     path = '../data/0HP'
#     train_X, train_Y, valid_X, valid_Y = prepro(d_path=path,
#                                 length=2048,
#                                 number=850,
#                                 normal=True,
#                                 enc=True,
#                                 enc_step=28,
#                                 snr=-2,
#                                 property='Train',
#                                 noise=True
#                                 )
#
#     test_X, test_Y = prepro(d_path=path,
#                              length=2048,
#                              number=150,
#                              normal=True,
#                              enc=True,
#                              enc_step=28,
#                              snr=-2,
#                              property='Test',
#                              noise=True
#                              )
#
#     # savemat('../data/0.1HP-1800_mat/data.mat',{'train_X': train_X,
#     #                                            'train_Y': train_Y,
#     #                                            'test_X': test_X,
#     #                                            'test_Y': test_Y})
#     train_X, valid_X, test_X = train_X[:, np.newaxis, :], valid_X[:, np.newaxis, :], test_X[:, np.newaxis, :]
#
#
#
#
#     pass





from scipy.io import loadmat
import numpy as np
import os
from sklearn import preprocessing  # 0-1编码
from sklearn.model_selection import StratifiedShuffleSplit  # 随机划分，保证每一类比例相同


def prepro(d_path, length=864, number=1000, normal=True, rate=[0.5, 0.25, 0.25], enc=True, enc_step=28, noise=True, snr=-2):
    """对数据进行预处理,返回train_X, train_Y, valid_X, valid_Y, test_X, test_Y样本.

    :param d_path: 源数据地址
    :param length: 信号长度，默认2个信号周期，864
    :param number: 每种信号个数,总共10类,默认每个类别1000个数据
    :param normal: 是否标准化.True,Fales.默认True
    :param rate: 训练集/验证集/测试集比例.默认[0.5,0.25,0.25],相加要等于1
    :param enc: 训练集、验证集是否采用数据增强.Bool,默认True
    :param enc_step: 增强数据集采样顺延间隔
    :return: Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y

    ```
    import preprocess.preprocess_nonoise as pre

    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = pre.prepro(d_path=path,
                                                                    length=864,
                                                                    number=1000,
                                                                    normal=False,
                                                                    rate=[0.5, 0.25, 0.25],
                                                                    enc=True,
                                                                    enc_step=28)
    ```
    """
    # 获得该文件夹下所有.mat文件名
    filenames = os.listdir(d_path)

    def capture(original_path):
        """读取mat文件，返回字典

        :param original_path: 读取路径
        :return: 数据字典
        """
        files = {}
        for i in filenames:
            # 文件路径
            file_path = os.path.join(d_path, i)
            file = loadmat(file_path)
            file_keys = file.keys()
            for key in file_keys:
                if 'DE' in key:
                    files[i] = file[key].ravel()
        return files

    # 江南大学数据集
    # def capture(original_path):
    #     files = {}
    #     for i in filenames:
    #         file_path = os.path.join(d_path, i)
    #         file = loadmat(file_path)
    #         file_keys = file.keys()
    #         for key in file_keys:
    #             if 'data' in key:
    #                 files[i] = file[key].ravel()
    #     return files

    # 东南大学数据集
    # def capture(original_path):
    #     files = {}
    #     for i in filenames:
    #         file_path = os.path.join(d_path, i)
    #         file = loadmat(file_path)
    #         file_keys = file.keys()
    #         for key in file_keys:
    #             if 'dndata' in key:
    #                 files[i] = file[key].ravel()
    #     # print(files)
    #     return files

    def slice_enc(data, slice_rate= rate[2]):
        """将数据切分为前面多少比例，后面多少比例.

        :param data: 单挑数据
        :param slice_rate: 验证集以及测试集所占的比例
        :return: 切分好的数据
        """
        keys = data.keys()
        Train_Samples = {}
        Test_Samples = {}
        for i in keys:
            slice_data = data[i]
            all_lenght = len(slice_data)
            end_index = int(all_lenght * (1 - slice_rate))
            samp_train = int(number * (1 - slice_rate))  # 700
            Train_sample = []
            Test_Sample = []
            if enc:
                enc_time = length // enc_step
                samp_step = 0  # 用来计数Train采样次数
                for j in range(samp_train):
                    random_start = np.random.randint(low=0, high=(end_index - 2 * length))
                    label = 0
                    for h in range(enc_time):
                        samp_step += 1
                        random_start += enc_step
                        sample = slice_data[random_start: random_start + length]
                        Train_sample.append(sample)
                        if samp_step == samp_train:
                            label = 1
                            break
                    if label:
                        break
            else:
                for j in range(samp_train):
                    random_start = np.random.randint(low=0, high=(end_index - length))
                    sample = slice_data[random_start:random_start + length]
                    Train_sample.append(sample)

            # 抓取测试数据
            for h in range(number - samp_train):
                random_start = np.random.randint(low=end_index, high=(all_lenght - length))
                sample = slice_data[random_start:random_start + length]
                Test_Sample.append(sample)
            Train_Samples[i] = Train_sample
            Test_Samples[i] = Test_Sample
        return Train_Samples, Test_Samples

    # 仅抽样完成，打标签
    def add_labels(train_test):
        X = []
        Y = []
        label = 0
        for i in filenames:
            x = train_test[i]
            X += x
            lenx = len(x)
            Y += [label] * lenx
            label += 1
        return X, Y

    # one-hot编码
    def one_hot(Train_Y, Test_Y):
        Train_Y = np.array(Train_Y).reshape([-1, 1])
        Test_Y = np.array(Test_Y).reshape([-1, 1])
        Encoder = preprocessing.OneHotEncoder()
        Encoder.fit(Train_Y)
        Train_Y = Encoder.transform(Train_Y).toarray()
        Test_Y = Encoder.transform(Test_Y).toarray()
        Train_Y = Encoder.inverse_transform(Train_Y)
        Test_Y = Encoder.inverse_transform(Test_Y)
        Train_Y = np.asarray(Train_Y, dtype=np.int32)
        Test_Y = np.asarray(Test_Y, dtype=np.int32)
        return Train_Y, Test_Y

    def scalar_stand(Train_X, Test_X):
        # 用训练集标准差标准化训练集以及测试集
        scalar = preprocessing.StandardScaler().fit(Train_X)
        Train_X = scalar.transform(Train_X)
        Test_X = scalar.transform(Test_X)
        return Train_X, Test_X

    # def valid_test_slice(Test_X, Test_Y):
    #     test_size = rate[2] / (rate[1] + rate[2])
    #     ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
    #     for train_index, test_index in ss.split(Test_X, Test_Y):
    #         X_valid, X_test = Test_X[train_index], Test_X[test_index]
    #         Y_valid, Y_test = Test_Y[train_index], Test_Y[test_index]
    #         return X_valid, Y_valid, X_test, Y_test

    def valid_test_slice(Train_X, Train_Y):
        train_size = 3 / 17
        ss = StratifiedShuffleSplit(n_splits=1, test_size=train_size)
        for train_index, test_index in ss.split(Train_X, Train_Y):
            X_valid, X_test = Train_X[train_index], Train_X[test_index]
            Y_valid, Y_test = Train_Y[train_index], Train_Y[test_index]
            return X_valid, Y_valid, X_test, Y_test

    # 高斯白噪声
    # def wgn(x, snr):
    #     snr = 10 ** (snr / 10.0)  # 将SNR从dB转换为线性标度
    #     xpower = np.sum(np.absolute(x) ** 2, axis=0) / x.shape[0]  # 计算平均信号功率
    #     npower = xpower / snr  # 根据信噪比计算噪声功率
    #     return np.random.standard_normal(x.shape) * np.sqrt(npower)  # 生成并返回具有计算功率的高斯白噪声
    #
    # def add_noise(data, snr_num):
    #     """Add Gaussian white noise to input data."""
    #     noisy_data = []
    #     for sample in data:
    #         rand_data = wgn(sample, snr_num)
    #         noisy_sample = sample + rand_data
    #         noisy_data.append(noisy_sample)
    #     return np.array(noisy_data)


    # 这个这个这个
    def wgn(x, snr):
        snr = 10 ** (snr / 10.0)  # 将SNR从dB转换为线性标度
        xpower = np.sum(np.absolute(x) ** 2, axis=0) / x.shape[0]  # 计算平均信号功率
        npower = xpower / snr  # 根据信噪比计算噪声功率
        return np.random.standard_normal(x.shape) * np.sqrt(npower)  # 生成并返回具有计算功率的高斯白噪声

        # 根据指定的 SNR 将高斯白噪声 (WGN) 添加到输入数据中
    def add_noise(data, snr_num):
        rand_data = wgn(data, snr_num)  # 生成具有指定SNR的高斯白噪声
        n_data = data + rand_data  # 将生成的噪声添加到输入数据
        return n_data  # 返回有噪声的数据和生成的噪声

    # def wgn(x, snr):
    #     """计算信噪比函数"""
    #     snr = 10 ** (snr / 10)
    #     xpower = np.sum(x ** 2) / len(x)
    #     npower = xpower / snr
    #     noise = np.random.randn(len(x)) * np.sqrt(npower)
    #     y = x + noise
    #     return y
    #
    # def add_noise(data, snr, length):
    #     """添加噪声函数"""
    #     data_noise = np.zeros((0, length))
    #     for i in data:
    #         b = wgn(i, snr)
    #         #         plt.plot(i)
    #         #         plt.plot(b)
    #         #         plt.show()
    #         b = b.reshape(-1, length)
    #         data_noise = np.vstack((data_noise, b))
    #     return data_noise

    # 粉红噪声
    # def pink_noise(shape, snr):
    #     snr_linear = 10 ** (snr / 10.0)  # Convert SNR from dB to linear scale
    #     pink_data = np.random.randn(*shape)  # Gaussian white noise
    #     pink_data = pink_data / np.linalg.norm(pink_data)  # Normalize
    #     pink_data = np.cumsum(pink_data)  # Cumulative sum to create pink noise
    #     pink_data = pink_data / np.max(np.abs(pink_data))  # Normalize to [-1, 1]
    #     pink_data = pink_data / np.sqrt(snr_linear)  # Adjust SNR
    #     return pink_data
    #
    # def add_pink_noise(data, snr_num):
    #     noisy_data = []
    #     for sample in data:
    #         shape = sample.shape
    #         pink_data = pink_noise(shape, snr_num)
    #         noisy_sample = sample + pink_data
    #         noisy_data.append(noisy_sample)
    #     return np.array(noisy_data)
    # def add_pink_noise(data, snr_num):
    #     noisy_data = []
    #     for sample in data:
    #         shape = sample.shape
    #         pink_data = pink_noise(shape, snr_num)
    #         noisy_sample = sample + pink_data
    #         noisy_data.append(noisy_sample)
    #     return np.array(noisy_data)

    # 拉普拉斯噪声
    # def laplace_noise(shape, snr):
    #     snr_linear = 10 ** (snr / 10.0)  # Convert SNR from dB to linear scale
    #     laplace_data = np.random.laplace(0, 1, shape)  # Laplace noise
    #     laplace_data = laplace_data / np.std(laplace_data)  # Normalize
    #     laplace_data = laplace_data / np.sqrt(snr_linear)  # Adjust SNR
    #     return laplace_data
    #
    # def add_laplace_noise(data, snr_num):
    #     laplace_data = laplace_noise(data, snr_num)
    #     noisy_data = data + laplace_data
    #     return noisy_data
    # def add_laplace_noise(data, snr_num):
    #     noisy_data = []
    #     for sample in data:
    #         shape = sample.shape
    #         laplace_data = laplace_noise(shape, snr_num)
    #         noisy_sample = sample + laplace_data
    #         noisy_data.append(noisy_sample)
    #     return np.array(noisy_data)

    # 从所有.mat文件中读取出数据的字典
    data = capture(original_path=d_path)
    # 将数据切分为训练集、测试集
    train, test = slice_enc(data)
    # 为训练集制作标签，返回X，Y
    Train_X, Train_Y = add_labels(train)
    # 为测试集制作标签，返回X，Y
    Test_X, Test_Y = add_labels(test)
    # 为训练集Y/测试集One-hot标签
    Train_Y, Test_Y = one_hot(Train_Y, Test_Y)
    Train_X = np.asarray(Train_X)
    Test_X = np.asarray(Test_X)
    plt.figure(1, figsize=(30, 2.5))
    plt.plot(Train_X[1])
    plt.tight_layout()
    plt.show()
    if noise == True:
        # Train_X = add_noise(Train_X, snr, length)
        # Test_X = add_noise(Test_X, snr, length)
        Train_X = add_noise(Train_X, snr)
        Test_X = add_noise(Test_X, snr)
        noise = wgn(Train_X, snr)
        # Train_X = add_laplace_noise(Train_X, snr)
        # Test_X = add_laplace_noise(Test_X, snr)
        plt.figure(2, figsize=(30, 2.5))
        plt.plot(Train_X[1])
        plt.tight_layout()
        plt.show()
        plt.figure(3, figsize=(30, 2.5))
        plt.plot(noise[1])
        plt.tight_layout()
        plt.show()
    # 训练数据/测试数据 是否标准化.
    if normal:
        Train_X, Test_X = scalar_stand(Train_X, Test_X)
    # else:
    #     # 需要做一个数据转换，转换成np格式.
    #     Train_X = np.asarray(Train_X)
    #     Test_X = np.asarray(Test_X)
    # 将测试集切分为验证集合和测试集.
    # Valid_X, Valid_Y, Test_X, Test_Y = valid_test_slice(Test_X, Test_Y)
    Train_X, Train_Y, Valid_X, Valid_Y = valid_test_slice(Train_X, Train_Y)
    return Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y


if __name__ == "__main__":
    path = '../data/0HP'
    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = prepro(d_path=path,
                                                                length=2048,
                                                                number=1000,
                                                                normal=False,
                                                                rate=[0.7, 0.15, 0.15],
                                                                enc=True,
                                                                enc_step=28,
                                                                noise=True,
                                                                snr=-2
                                                                )
