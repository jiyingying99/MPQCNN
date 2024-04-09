import os
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import random
from torch import nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, recall_score, \
    precision_score, f1_score
import itertools
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from Model.mymodel1_4 import mymodel1_4
from Model.WDCNN import WDCNN
from Model.QCNN import QCNN
from utils.Preprocess import prepro
from utils.DatasetLoader import CustomTensorDataset
from utils.train_function import group_parameters
from utils.label_smoothing import LSR
from utils.AdamP_amsgrad import AdamP
from Model.DCABiGRU import Net1
from Model.EWSNet import Net

import torch.optim as optim

# cpu or gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()
# device = torch.device("cuda:1")

plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置中文字体为SimHei
plt.rcParams['font.size'] = 10  # 设置字体大小为10号
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.size'] = 10
# plt.rc('font', family='Times New Roman')
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

#随机种子
def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def select_model(chosen_model):
    if chosen_model == 'wdcnn':
        model = WDCNN()
    if chosen_model == 'qcnn':
        model = QCNN()
    if chosen_model == 'mymodel1_4':
        model = mymodel1_4()
    if chosen_model == 'dca':
        model = Net1()
    if chosen_model == 'ews':
        model = Net()
    return model

total_train_loss = []
total_test_loss = []
total_train_acc = []
total_test_acc = []
total_valid_loss = []
total_valid_acc = []
y_list, y_predict_list = [], []
predictions = []



def train(chosen_model, dataloader, epochs, lr, alpha):
    net = select_model(chosen_model)
    if use_gpu:
        net.cuda()
    for e in range(epochs):
        for phase in ['train', 'validation']:
            loss = 0
            total = 0
            correct = 0
            loss_total = 0
            epoch_loss = 0

            if phase == 'train':
                net.train()
            if phase == 'validation':
                net.eval()
                torch.no_grad()

            start_time = time.time()

            for step, (x, y) in enumerate(dataloader[phase]):

                x = x.type(torch.float)
                y = y.type(torch.long)
                y = y.view(-1)
                if use_gpu:
                    x, y = x.cuda(), y.cuda()
                if chosen_model == 'mymodel1_4':
                    group = group_parameters(net)
                    optimizer = torch.optim.SGD([
                        {"params": group[0], "lr": lr},  # weight_r
                        {"params": group[1], "lr": lr * alpha},  # weight_g
                        {"params": group[2], "lr": lr * alpha},  # weight_b
                        {"params": group[3], "lr": lr},  # bias_r
                        {"params": group[4], "lr": lr * alpha},  # bias_g
                        {"params": group[5], "lr": lr * alpha},  # bias_b
                        {"params": group[6], "lr": lr},
                        {"params": group[7], "lr": lr},
                    ], lr=lr, momentum=0.9, weight_decay=1e-4)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 0.01)
                    # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,
                                                                        eta_min=1e-8)  # goal: maximize Dice score

                else:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 0.01)
                    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,
                                                                            eta_min=1e-8)
                # loss_func = nn.CrossEntropyLoss()
                loss_func = LSR()
                if use_gpu:
                    y_hat = net(x).cuda()       # y_hat shape:torch.size([64,10])
                else:
                    y_hat = net(x)
                loss = loss_func(y_hat, y)
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                loss_total += loss.item()       # 将每次迭代中的损失值加总到loss_total变量中，loss.item()用于获取损失张量中的数值
                epoch_loss = loss.item()
                y_predict = y_hat.argmax(dim=1)

                total += y.size(0)
                if use_gpu:
                    # correct += (y_predict == y).cpu().squeeze().sum().numpy()
                    correct += (y_predict == y).cpu().sum().item()
                else:
                    correct += (y_predict == y).squeeze().sum().numpy()
                # correct += (y_predict == y).to(device).squeeze().sum().numpy()
                if step % 20 == 0 and phase == 'train':
                    print('Epoch:%d, Step [%d/%d], Loss: %.4f'
                          % (
                          e + 1, step + 1, len(dataloader[phase].dataset), epoch_loss))
            # loss_total = loss_total / len(dataloader[phase].dataset)

            acc = correct / total
            # epoch_loss = loss_total / total
            if phase == 'train':
                total_train_loss.append(epoch_loss)
                total_train_acc.append(acc)
                print('%s ACC:%.4f' % (phase, acc))
                print('%s LOSS:%.4f' % (phase, epoch_loss))
            if phase == 'validation':
                scheduler.step(epoch_loss)
                total_valid_loss.append(epoch_loss)
                total_valid_acc.append(acc)
                print('%s ACC:%.4f' % (phase, acc))
                print('%s LOSS:%.4f' % (phase, epoch_loss))
            end_time = time.time()
            epoch_time = end_time - start_time
            print(f"Time taken for one epoch: {epoch_time} seconds")

    plt.figure(1)
    plt.plot(range(epochs), total_train_acc, label='Train Accuracy')
    plt.plot(range(epochs), total_valid_acc, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    # plt.title('Accurancy')
    plt.legend()
    # plt.savefig('./pic/accuracy.jpg', dpi=500)

    plt.figure(2)
    plt.plot(range(epochs), total_train_loss, label='Train Loss')
    plt.plot(range(epochs), total_valid_loss, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.title('Loss')
    plt.legend()
    # plt.savefig('./pic/MPQCNN_loss.jpg', dpi=500)
    plt.show()
    return net




def inference(dataloader, model):
    net = model
    if use_gpu:
        net.cuda()
    net.eval()
    loss = 0
    loss_total = 0
    total = 0
    correct = 0
    # endregion
    with torch.no_grad():
        for step, (x, y) in enumerate(dataloader):
            x = x.type(torch.float)
            y = y.type(torch.long)
            y = y.view(-1)
            if use_gpu:
                x, y = x.cuda(), y.cuda()
            y_hat = net(x)
            loss_func = LSR()
            # loss_func = nn.CrossEntropyLoss()
            loss = loss_func(y_hat, y)
            loss_total = loss.item()
            y_predict = y_hat.argmax(dim=1)
            total += y.size(0)
            correct += (y_predict == y).cpu().squeeze().sum().numpy()
            y_list.extend(y.detach().cpu().numpy())
            y_predict_list.extend(y_predict.detach().cpu().numpy())
            predictions.extend(y_hat.cpu().numpy())
        acc = correct / total
        print('test ACC:%.4f' % (acc))
        print('test LOSS:%.4f' % (loss_total))

        cnf_matrix = confusion_matrix(y_list, y_predict_list)
        recall = recall_score(y_list, y_predict_list, average="macro")
        precision = precision_score(y_list, y_predict_list, average="macro")

        F1 = f1_score(y_list, y_predict_list, average="macro")
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)

        FP = FP.astype(float)
        TN = TN.astype(float)
        FPR = np.mean(FP / (FP + TN))

        save_path = "net/checkpoint.pth"
        torch.save(net.state_dict(), save_path)
        print('model saved')

        # 绘制混淆矩阵
        plot_confusion_matrix(cnf_matrix, classes=range(10))

        plt.xlabel('Predict Label')
        plt.ylabel('True Label')
        # plt.title('')
        plt.tight_layout()  # 自动调整各子图间距
        # plt.savefig('./pic/MPQCNN.jpg', dpi=500)
        plt.show()

        # Convert predictions to numpy array for t-SNE
        predict_data = np.array(predictions)

        # Compute t-SNE embeddings
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        tsne_embeddings = tsne.fit_transform(predict_data)

        # Plot t-SNE embeddings
        plot_embedding(tsne_embeddings, 'T-SNE Embedding Of A_Data', y_list)

        plt.tight_layout()
        # plt.savefig('./pic/matrix_MPQCNN111.jpg', dpi=500)
        plt.show()


        return F1

def plot_embedding(X, title, y):
    # Create a scatter plot of the t-SNE embeddings
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, label=y)  # 使用预测类别作为标签
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(*scatter.legend_elements(), bbox_to_anchor=(1.05, 1), loc='upper left')  # 添加图例并设置位置
    plt.axis('equal')  # 设置坐标轴比例相等
def plot_confusion_matrix(cm, classes, cmap=plt.cm.Blues, normalize=False):
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    tick_mark = np.arange(len(classes))
    plt.xticks(tick_mark, classes, rotation=0)
    plt.yticks(tick_mark, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = '%.2f'%cm
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if i == j:
            plt.text(j, i, cm[i, j], horizontalalignment='center', color='white', fontsize=10)
        else:
            plt.text(j, i, cm[i, j], horizontalalignment='center', color='black', fontsize=10)
    # plt.tight_layout()
    # plt.ylabel('真实标签')
    # plt.xlabel('预测标签')


def main():
    device = torch.device("cuda")
    random_seed(seed)

    path = os.path.join('data', chosen_data)

    train_X, train_Y, valid_X, valid_Y, test_X, test_Y =prepro(d_path=path,
                                                               length=2048,
                                                               number=1000,
                                                               normal=True,
                                                               rate=[0.7, 0.15, 0.15],
                                                               enc=True,
                                                               enc_step=28,
                                                               noise=add_noise,
                                                               snr=snr
                                                               )

    train_X, valid_X, test_X = train_X[:, np.newaxis, :], valid_X[:, np.newaxis, :], test_X[:, np.newaxis, :]
    train_dataset = CustomTensorDataset(torch.tensor(train_X, dtype=torch.float), torch.tensor(train_Y))
    valid_dataset = CustomTensorDataset(torch.tensor(valid_X, dtype=torch.float), torch.tensor(valid_Y))
    test_dataset = CustomTensorDataset(torch.tensor(test_X, dtype=torch.float), torch.tensor(test_Y))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    data_loaders = {
        "train": train_loader,
        "validation": valid_loader
    }

    net = train(chosen_model, data_loaders, epochs, lr, alpha)
    f1 = inference(test_loader, net)
    print(f1)

if __name__ == '__main__':
    seed = 42

    batch_size = 64
    epochs = 100
    lr = 0.05
    # lr = 0.0001
    alpha = 0.03
    snr = -4
    add_noise = False

    # chosen_data = '20c2'
    # chosen_data = '30c2'
    # chosen_data = '30c4'
    # chosen_data = '600'
    chosen_data = '0HP'


    # 选择模型
    chosen_model = 'mymodel1_4'
    # chosen_model = 'wdcnn'
    # chosen_model = 'qcnn'
    # chosen_model = 'ews'
    # chosen_model = 'dca'

    main()