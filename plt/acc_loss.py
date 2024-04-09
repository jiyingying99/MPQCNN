from matplotlib import pyplot as plt
import numpy as np

epochs = 50

train_acc  = np.load('../acc/00qcnn_train_acc.npy')
val_acc = np.load('../acc/00qcnn_val_acc.npy')

train_loss  = np.load('../acc/00qcnn_train_loss.npy')
val_loss = np.load('../acc/00qcnn_val_loss.npy')

plt.figure(1)
plt.plot(range(epochs), train_acc, label='Train Accurancy')
plt.plot(range(epochs), val_acc, label='Valid Accurancy')
plt.xlabel('Epoch')
plt.ylabel('Accurancy')
plt.title('CWRU-Accurancy')
plt.legend()
# plt.savefig('output/ResNet18-CIFAR10-Accurancy.jpg')  # 自动保存plot出来的图片

plt.figure(2)
plt.plot(range(epochs), train_loss, label='Train Loss')
plt.plot(range(epochs), val_loss, label='Valid Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('CWRU-Loss')
plt.legend()
# plt.savefig('output/ResNet18-CIFAR10-Accurancy.jpg')  # 自动保存plot出来的图片
plt.show()