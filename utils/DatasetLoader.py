import torch
from torch.utils.data import Dataset

# 处理包含张量数据的数据集，并支持数据转换操作
class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    # 初始化方法，初始化数据集 X 和标签集 y。还有一个参数 transform，用于数据转换。
    def __init__(self, X, y, transform=None):
        # assert all(X.size(0) == tensor.size(0) for tensor in X)
        # assert all(y.size(0) == tensor.size(0) for tensor in y)

        self.X = X
        self.y = y

        self.transform = transform

    # 用于获取索引 index 处的数据，其中应用了可能存在的数据转换。
    def __getitem__(self, index):
        x = self.X[index]

        if self.transform:
            x = self.transform(x)

        y = self.y[index]

        return x, y

    # 用于获取数据集的长度
    def __len__(self):
        return len(self.X)

