import torch
import torch.nn as nn
# from wmodelsii8 import Sin_fast as fast
# from wmodelsii3 import Laplace_fast as fast
from Model.lapeplace import Laplace_fastv2 as fast
# from wsinc import SincConv_fast as fast
from Model.Shrinkage import Shrinkage as sage
from fvcore.nn import FlopCountAnalysis, flop_count_str
import torch.nn.functional as F
from torchsummary import summary
from Model.CBAM import CBAM

# adam lr_scheduler.StepLR lsr 2048 700:300 normal enc28 batch_size = 16 lr = 0.001
# 0hp
# 0.7:0.15:0.15 样本量每种1000 none:1.0000(1.0)
# -6:0.5727(0.5402521953466464)  -4:0.8513(0.8482606274266924)  snr-2:0.9240(0.9248713175317324)  0:0.9527(0.9518941812322337)
# 2:0.9867(0.9865233969948616)  4:0.9927(0.9925978518594537)  6:1.0000(1.0)  8:1.0000(1.0)  10:1.0000(1.0)
# 0-1:0.6567(0.6116074412387211)  0-2:0.5073(0.43970434182512913)  0-3:0.4933(0.41708880390042397)

# cwru 0hp  adam lr_scheduler.StepLR lsr 2048 700:300 normal enc28 batch_size = 16 lr = 0.001
# 0.7:0.15:0.15  样本量每种1000 none:1.0000(1.0)
# 0-1:0.7553(0.703761954450315) 0-2:0.5753(0.4957807083837135) 0-3:0.4587(0.3953035064906797)
# -4:0.7987(0.8018612410504057)
class Mish1(nn.Module):
    def __init__(self):
        super(Mish1, self).__init__()
        self.mish = nn.ReLU(inplace=True)

    def forward(self, x):

        return self.mish(x)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()    #85,42,70   #63,31,75
        self.p1_0 = nn.Sequential(  # nn.Conv1d(1, 50, kernel_size=18, stride=2),
            # fast(out_channels=64, kernel_size=250, stride=1),
            # fast1(out_channels=70, kernel_size=84, stride=1),
            nn.Conv1d(1, 64, kernel_size=250, stride=1, bias=True),
            nn.BatchNorm1d(64),
            Mish1()
        )           # ,64,1799
        self.p1_1 = nn.Sequential(nn.Conv1d(64, 16, kernel_size=18, stride=2, bias=True),
                                  # fast(out_channels=50, kernel_size=18, stride=2),
                                  nn.BatchNorm1d(16),
                                  Mish1()
                                  )         # , 16,891
        self.p1_2 = nn.Sequential(nn.Conv1d(16, 10, kernel_size=10, stride=2, bias=True),
                                  nn.BatchNorm1d(10),
                                  Mish1()
                                  )         # , 10,441
        self.p1_3 = nn.MaxPool1d(kernel_size=2)         # ,10,220
        self.p2_1 = nn.Sequential(nn.Conv1d(64, 32, kernel_size=6, stride=1, bias=True),
                                  # fast(out_channels=50, kernel_size=6, stride=1),
                                  nn.BatchNorm1d(32),
                                  Mish1()
                                  )         # , 32,1794
        self.p2_2 = nn.Sequential(nn.Conv1d(32, 16, kernel_size=6, stride=1, bias=True),
                                  nn.BatchNorm1d(16),
                                  Mish1()
                                  )         # ,16,1789
        self.p2_3 = nn.MaxPool1d(kernel_size=2)         # ,16,894
        self.p2_4 = nn.Sequential(nn.Conv1d(16, 10, kernel_size=6, stride=1, bias=True),
                                  nn.BatchNorm1d(10),
                                  Mish1()
                                  )         # ,10,889
        self.p2_5 = nn.Sequential(nn.Conv1d(10, 10, kernel_size=8, stride=2, bias=True),
                                  # nn.Conv1d(10, 10, kernel_size=6, stride=2),
                                  nn.BatchNorm1d(10),
                                  Mish1()
                                 )  # PRelu         # ,10,441
        self.p2_6 = nn.MaxPool1d(kernel_size=2)         # ,10,220
        self.p3_0 = sage(channel=64, gap_size=1)            # ,64,1799
        self.p3_1 = nn.Sequential(nn.Conv1d(64, 10, kernel_size=43, stride=4, bias=True),
                                  nn.BatchNorm1d(10),
                                  Mish1()
                                 )          # ,10,440
        self.p3_2 = nn.MaxPool1d(kernel_size=2)         # ,10,220
        self.p3_3 = nn.Sequential(nn.AdaptiveAvgPool1d(1))
        self.p4 = nn.Sequential(nn.Linear(10, 5))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                if m.kernel_size == (500,):
                    m.weight.data = fast(out_channels=64, kernel_size=250).forward()
                    nn.init.constant_(m.bias.data, 0.0)
                else:
                    nn.init.kaiming_normal_(m.weight.data)
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.fill_(1)



    def forward(self, x):
        x = self.p1_0(x)
        # p1 = self.p1_1(x)
        # p1 = self.p1_2(p1)
        # p1 = self.p1_3(p1)
        # p2 = self.p2_1(x)
        # p2 = self.p2_2(p2)
        # p2 = self.p2_3(p2)
        # p2 = self.p2_4(p2)
        # p2 = self.p2_5(p2)
        # p2 = self.p2_6(p2)
        # p3 = self.p3_0(x)
        # p3 = self.p3_1(p3)
        # p3 = self.p3_2(p3)
        p1 = self.p1_3(self.p1_2(self.p1_1(x)))
        p2 = self.p2_6(self.p2_5(self.p2_4(self.p2_3(self.p2_2(self.p2_1(x))))))
        # out = torch.cat((p1, p2), dim=2)
        x = self.p3_2(self.p3_1(x + self.p3_0(x)))
        # x = torch.cat((out, x), dim=2)
        x = torch.add(x, torch.add(p1, p2))
        x = self.p3_3(x).squeeze()
        x = self.p4(x)
        return x

if __name__ == '__main__':
    X = torch.rand(2, 1, 1024).cuda()
    m = Net()
    summary(m.cuda(), (1, 2048))
    print(flop_count_str(FlopCountAnalysis(m, X)))