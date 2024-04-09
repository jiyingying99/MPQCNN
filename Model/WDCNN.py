import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, flop_count_str
import torch.nn.functional as F
from Model.CBAM import CBAM


# snr-2:0.9100  0:0.9600    0.9987  0->1:0.8060 0->2:0.8087 0->3:0.8133
class WDCNN(nn.Module):
    """
    WDCNN builder
    """

    def __init__(self, ) -> object:
        super(WDCNN, self).__init__()
        self.cnn = nn.Sequential()
        self.cnn.add_module('Conv1D_1', nn.Conv1d(1, 16, 64, 8, 28))
        self.cnn.add_module('BN_1', nn.BatchNorm1d(16))
        self.cnn.add_module('Relu_1', nn.ReLU())
        # self.cnn.add_module('CBAM_1', CBAM(16))
        self.cnn.add_module('MAXPool_1', nn.MaxPool1d(2, 2))
        self.__make_layer(16, 32, 1, 2)
        self.__make_layer(32, 64, 1, 3)   #  改64
        self.__make_layer(64, 64, 1, 4)   #  改64
        self.__make_layer(64, 64, 0, 5)   #  改64
        self.__make_layer(64, 64, 0, 6)
        self.fc1 = nn.Linear(128, 100)
        self.relu1 = nn.ReLU()
        self.dp = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, 5)

    def __make_layer(self, in_channels, out_channels, padding, nb_patch):
        self.cnn.add_module('Conv1D_%d' % (nb_patch), nn.Conv1d(in_channels, out_channels, 3, 1, padding))
        self.cnn.add_module('BN_%d' % (nb_patch), nn.BatchNorm1d(out_channels))
        self.cnn.add_module('ReLu_%d' % (nb_patch), nn.ReLU())
        self.cnn.add_module('MAXPool_%d' % (nb_patch), nn.MaxPool1d(2, 2))

    def forward(self, x):
        out = self.cnn(x)
        out = self.fc1(out.view(x.size(0), -1))
        out = self.relu1(out)
        out = self.dp(out)
        out = self.fc2(out)
        return F.softmax(out, dim=1)
        # return out

# class WDCNN(nn.Module):
#     """
#     QCNN builder
#     """
#
#     def __init__(self, ) -> object:
#         super(WDCNN, self).__init__()
#         self.p1 = nn.Sequential(nn.Conv1d(1, 16, 64, 1, 1),
#                                 nn.BatchNorm1d(16),
#                                 nn.ReLU(),
#                                 nn.MaxPool1d(2, 2)
#                                 )
#         self.p2 = nn.Sequential(nn.Conv1d(16, 32, 3, 1, 1),
#                                 nn.BatchNorm1d(32),
#                                 nn.ReLU(),
#                                 nn.MaxPool1d(2, 2)
#                                 )
#         self.p3 = nn.Sequential(nn.Conv1d(32, 64, 3, 1, 1),
#                                 nn.BatchNorm1d(64),
#                                 nn.ReLU(),
#                                 nn.MaxPool1d(2, 2)
#                                 )
#         self.p4 = nn.Sequential(nn.Conv1d(64, 64, 3, 1, 1),
#                                 nn.BatchNorm1d(64),
#                                 nn.ReLU(),
#                                 nn.MaxPool1d(2, 2)
#                                 )
#         self.p5 = nn.Sequential(nn.Conv1d(64, 64, 3, 1, 1),
#                                 nn.BatchNorm1d(64),
#                                 nn.ReLU(),
#                                 nn.MaxPool1d(2, 2)
#                                 )
#         # self.p6 = nn.Sequential(nn.Conv1d(64, 64, 3, 1, 0),
#         #                         nn.BatchNorm1d(64),
#         #                         nn.ReLU(),
#         #                         nn.MaxPool1d(2, 2)
#         #                         )
#
#         self.fc1 = nn.Linear(3968, 100)
#         self.relu1 = nn.ReLU()
#         self.dp = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(100, 10)
#
#     def forward(self, x):
#         # out1 = self.cnn(x)
#         x = self.p1(x)
#         x = self.p2(x)
#         x = self.p3(x)
#         x = self.p4(x)
#         x = self.p5(x)
#         # out1 = self.p6(x)
#         out = self.fc1(x.view(x.size(0), -1))
#         out = self.relu1(out)
#         out = self.dp(out)
#         out = self.fc2(out)
#         # return F.softmax(out, dim=1)
#         return out


if __name__ == '__main__':
    X = torch.rand(1, 1, 2048)
    m = WDCNN()
    print(flop_count_str(FlopCountAnalysis(m, X)))