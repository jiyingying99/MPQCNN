import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, flop_count_str
from torchsummary import summary
import torch.nn.functional as F
from Model.ConvQuadraticOperation import ConvQuadraticOperation

# sgd 梯度剪裁 余弦退火 交叉熵 700:300 normal enc28 2048 lr0.05 batchsize64
# 0hp
# 0.7:0.15:0.15  none:0.9993(0.9993333259258437)
# snr-6:0.6467(0.6459861629683478)  -4:0.7633(0.7645699001499341)  -2:0.8567(0.8569873657596766)  0:0.9347(0.9348314087172263)
# 2:0.9720(0.971902601787938)  4:0.9893(0.9892978372665701)  6:0.9920(0.9920039784446564)  8:0.9980(0.9979999554059177)  10:1.0000(1.0)
# 0-1:0.9313(0.9264554419604245)  0-2:0.8687(0.8525411671617549)  0-3:0.8500(0.8176277914991621)
class QCNN(nn.Module):
    """
    QCNN builder
    """

    def __init__(self, ) -> object:
        super(QCNN, self).__init__()
        self.cnn = nn.Sequential()
        # self.cnn1 = nn.Sequential()
        self.cnn.add_module('Conv1D_1', ConvQuadraticOperation(1, 16, 64, 8, 28))
        # self.cnn.add_module('Conv1D_1', nn.Conv1d(1, 16, 64, 8, 28))
        self.cnn.add_module('BN_1', nn.BatchNorm1d(16))
        self.cnn.add_module('Relu_1', nn.ReLU())
        self.cnn.add_module('MAXPool_1', nn.MaxPool1d(2, 2))
        self.__make_layerq(16, 32, 1, 2)
        self.__make_layerq(32, 64, 1, 3)
        self.__make_layerq(64, 64, 1, 4)
        self.__make_layerq(64, 64, 1, 5)
        self.__make_layerq(64, 64, 0, 6)

        # self.__make_layerc(16, 32, 1, 2)
        # self.__make_layerc(32, 64, 1, 3)
        # self.__make_layerc(64, 64, 1, 4)
        # self.__make_layerc(64, 64, 1, 5)
        # self.__make_layerc(64, 64, 0, 6)
        self.fc1 = nn.Linear(192, 100)
        # self.fc1 = nn.Linear(448, 100)
        self.relu1 = nn.ReLU()
        self.dp = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, 5)



    def __make_layerq(self, in_channels, out_channels, padding, nb_patch):
        self.cnn.add_module('Conv1D_%d' % (nb_patch), ConvQuadraticOperation(in_channels, out_channels, 3, 1, padding))
        self.cnn.add_module('BN_%d' % (nb_patch), nn.BatchNorm1d(out_channels))
        # self.cnn.add_module('DP_%d' %(nb_patch), nn.Dropout(0.5))
        self.cnn.add_module('ReLu_%d' % (nb_patch), nn.ReLU())
        self.cnn.add_module('MAXPool_%d' % (nb_patch), nn.MaxPool1d(2, 2))

    def __make_layerc(self, in_channels, out_channels, padding, nb_patch):
        self.cnn.add_module('Conv1D_%d' % (nb_patch), nn.Conv1d(in_channels, out_channels, 3, 1, padding))
        self.cnn.add_module('BN_%d' % (nb_patch), nn.BatchNorm1d(out_channels))
        # self.cnn.add_module('DP_%d' %(nb_patch), nn.Dropout(0.5))
        self.cnn.add_module('ReLu_%d' % (nb_patch), nn.ReLU())
        self.cnn.add_module('MAXPool_%d' % (nb_patch), nn.MaxPool1d(2, 2))

    def forward(self, x):
        out1 = self.cnn(x)
        out = self.fc1(out1.view(x.size(0), -1))
        out = self.relu1(out)
        out = self.dp(out)
        out = self.fc2(out)
        return F.softmax(out, dim=1)


if __name__ == '__main__':
    X = torch.rand(1, 1, 4096).cuda()
    m = QCNN()
    summary(m.cuda(), (1, 4096))
    print(flop_count_str(FlopCountAnalysis(m.cuda(), X)))