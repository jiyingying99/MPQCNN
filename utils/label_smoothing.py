import torch
import torch.nn as nn


class LSR(nn.Module):

    def __init__(self, e=0.1, reduction='mean'):
        super().__init__()

        self.log_softmax = nn.LogSoftmax(dim=1)     # 计算 softmax 函数沿指定维度的对数。这通常用于获取分类问题中类的概率分布。
        self.e = e      # 损失函数的平滑参数
        self.reduction = reduction      # 缩减方法

    # 将标签转换为one-hot格式
    def _one_hot(self, labels, classes, value=1):
        """
            Convert labels to one hot vectors

        Args:
            labels: torch tensor in format [label1, label2, label3, ...]
            classes: int, number of classes
            value: label value in one hot vector, default to 1

        Returns:
            return one hot format labels in shape [batchsize, classes]
        """

        one_hot = torch.zeros(labels.size(0), classes)

        # labels and value_added  size must match
        labels = labels.view(labels.size(0), -1)        # 重塑labels形状
        value_added = torch.Tensor(labels.size(0), 1).fill_(value)      # 创建一个value_added具有与labels相同行数labels并用指定value填充的单列的张量。

        value_added = value_added.to(labels.device)
        one_hot = one_hot.to(labels.device)     # 移动one_hot张量到与labels张量相同的设备

        one_hot.scatter_add_(1, labels, value_added)        # 使用scatter_add_ 根据张量和张量指定的位置来更新张量

        return one_hot

    def _smooth_label(self, target, length, smooth_factor):
        """convert targets to one-hot format, and smooth
        them.
        Args:
            target: target in form with [label1, label2, label_batchsize]
            length: length of one-hot format(number of classes)
            smooth_factor: smooth factor for label smooth

        Returns:
            smoothed labels in one hot format
        """
        one_hot = self._one_hot(target, length, value=1 - smooth_factor)        # 将标签转换target为one-hot格式
        one_hot += smooth_factor / (length - 1)

        return one_hot.to(target.device)

    def forward(self, x, target):

        if x.size(0) != target.size(0):     # 检查x和target的批量大小是否匹配
            raise ValueError('Expected input batchsize ({}) to match target batch_size({})'
                             .format(x.size(0), target.size(0)))

        if x.dim() < 2:     # 检查输入张量是否至少有两个维度
            raise ValueError('Expected input tensor to have least 2 dimensions(got {})'
                             .format(x.size(0)))

        if x.dim() != 2:        # 检查输入张量是否恰好具有二维
            raise ValueError('Only 2 dimension tensor are implemented, (got {})'
                             .format(x.size()))

        smoothed_target = self._smooth_label(target, x.size(1), self.e)
        x = self.log_softmax(x)
        loss = torch.sum(- x * smoothed_target, dim=1)

        # 根据指定的缩减方式，返回损失
        # 如果采用还原模式'none'，该函数按原样返回损失
        if self.reduction == 'none':
            return loss
        # 如果采用还原模式'sum'，函数返回损失的总和
        elif self.reduction == 'sum':
            return torch.sum(loss)
        # 如果缩减模式为'mean'，则函数返回损失的平均值
        elif self.reduction == 'mean':
            return torch.mean(loss)

        else:
            raise ValueError('unrecognized option, expect reduction to be one of none, mean, sum')