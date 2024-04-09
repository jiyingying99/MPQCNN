import torch
from torch import nn
from torchsummary import summary
from fvcore.nn import FlopCountAnalysis, flop_count_str
import torch.nn.functional as F
from Model.ConvQuadraticOperation import ConvQuadraticOperation
from Model.ECA import ECA_block
from Model.Shrinkage import Shrinkage as sage
from Model.CausalDilationConv1d import CausalDilationConv1d
from Model.CBAM import CBAM

# 0.9573   snr -2 :0.9453(nn.CrossEntropyLoss());  snr -2 :0.9740(LSR())
# 0->1:0.7373   0->2:0.7660     0->3:0.8020

# 加p5  0->1:0.8160   0->2:0.8987     0->3:0.9307    snr-2:0.9753()  0.8760(enc=false)
# test 加噪 0->1:0.8267 0->2:0.8900

# no eca  snr-2:0.9920(enc=false)   0.9953(enc=true)    0->1:0.8093 0->2:0.8033 0->3:0.8993
# class mymodel1_4(nn.Module):
#     """
#     QCNN builder
#     """
#
#     def __init__(self, ) -> object:
#         super(mymodel1_4, self).__init__()
#         self.p1 = nn.Sequential(ConvQuadraticOperation(1, 16, kernel_size=64, stride=8, padding=28),
#                                 # nn.Conv1d(1, 16, 3, 1, 1),
#                                 nn.BatchNorm1d(16),
#                                 nn.ReLU(),
#                                 nn.MaxPool1d(2, 2)
#                                 )
#
#         self.p1_1 = nn.Sequential(ConvQuadraticOperation(16, 32, kernel_size=3, stride=1, padding=1),
#                                 # nn.Conv1d(16, 32, 3, 1, 1),
#                                 nn.BatchNorm1d(32),
#                                 nn.ReLU(),
#                                 nn.MaxPool1d(2, 2)
#                                 )
#         self.p1_2 = nn.Sequential(ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#                                 # nn.Conv1d(32, 64, 3, 1, 1),
#                                 nn.BatchNorm1d(64),
#                                 nn.ReLU(),
#                                 nn.MaxPool1d(2, 2)
#                                 )
#         self.p1_3 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#                                 # nn.Conv1d(64, 64, 3, 1, 1),
#                                 nn.BatchNorm1d(64),
#                                 nn.ReLU(),
#                                 nn.MaxPool1d(2, 2)
#                                 )
#         # 阈值降噪
#         self.p2_1 = sage(channel=64, gap_size=1)  # , 64, 1551
#         self.p2_2 = ECA_block(64)
#         self.p2_3 = nn.Sequential(ConvQuadraticOperation(64, 64, 43, 4, 18),
#                                   # nn.Conv1d(64, 32, 43, 4, bias=True),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU()
#                                   )         # , 32, 378
#         self.p2_4 = nn.MaxPool1d(2)         # , 32, 189
#
#         self.p3_1 = nn.Sequential(CausalDilationConv1d(16, 32, kernel_size=3, stride=1, dilation=5),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=1),
#                                   nn.BatchNorm1d(32),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p3_2 = nn.Sequential(ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=3),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU()
#                                   )
#         self.p3_3 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=3),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU()
#                                   )
#         self.p3_4 = nn.MaxPool1d(2)
#         self.p4_1 = nn.Sequential(CausalDilationConv1d(16, 32, kernel_size=3, stride=1, dilation=3),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=3),
#                                   nn.BatchNorm1d(32),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p4_2 = nn.Sequential(ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=3),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   )
#         self.p4_3 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=3),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU()
#                                   )
#         self.p4_4 = nn.MaxPool1d(2)
#         self.p5_1 = nn.Sequential(CausalDilationConv1d(16, 32, kernel_size=3, stride=1, dilation=1),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=3),
#                                   nn.BatchNorm1d(32),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p5_2 = nn.Sequential(ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=3),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   )
#         self.p5_3 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=3),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU()
#                                   )
#         self.p5_4 = nn.MaxPool1d(2)
#         # self
#         self.fc1 = nn.Linear(6208, 100)
#         self.relu1 = nn.ReLU()
#         self.dp = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(100, 10)
#
#     def forward(self, x):
#         x = self.p1(x)
#         p3 = self.p3_4(self.p3_3(self.p3_2(self.p3_1(x))))
#         p4 = self.p4_4(self.p4_3(self.p4_2(self.p4_1(x))))
#         p5 = self.p5_4(self.p5_3(self.p5_2(self.p5_1(x))))
#         p1 = self.p1_1(x)
#         p1 = self.p1_2(p1)
#         p1 = self.p1_3(p1)
#         p1out = self.p2_4(self.p2_3(self.p2_2(p1 + self.p2_1(p1))))
#         # p1out = self.p2_4(self.p2_3(p1 + self.p2_1(p1)))
#         out1 = torch.cat((p1out, torch.cat((p5, torch.cat((p3, p4), dim=2)), dim=2)), dim=2)
#         out = self.fc1(out1.view(x.size(0), -1))
#         out = self.relu1(out)
#         out = self.dp(out)
#         out = self.fc2(out)
#         # return F.softmax(out, dim=1)
#         return out



# 1.0: q+阈值(no eca enc=true)out -2：0.9613            0->1:0.8580   0->2:0.8280   0->3:0.8627
# 1.1：同1.0 有eca -2:0.9947 -4:0.7290  -2:0.9960(enc=false) -4:0.8467         0->1:0.8833   0->2:0.8460   0->3:0.8080(选的有enc的)
# 去掉64-64 0->1:0.8973  0->2:0.9240  0->3:0.9200   snr-2:0.9387(enc=false) 上采样1
# class mymodel1_4(nn.Module):
#     """
#     QCNN builder
#     """
#
#     def __init__(self, ) -> object:
#         super(mymodel1_4, self).__init__()
#         self.p1 = nn.Sequential(ConvQuadraticOperation(1, 16, kernel_size=64, stride=8, padding=28),
#                                 # nn.Conv1d(1, 16, 3, 1, 1),
#                                 nn.BatchNorm1d(16),
#                                 nn.ReLU(),
#                                 nn.MaxPool1d(2, 2)
#                                 )
#
#         self.p1_1 = nn.Sequential(ConvQuadraticOperation(16, 32, kernel_size=3, stride=1, padding=1),
#                                 # nn.Conv1d(16, 32, 3, 1, 1),
#                                 nn.BatchNorm1d(32),
#                                 nn.ReLU(),
#                                 nn.MaxPool1d(2, 2)
#                                 )
#         self.p1_2 = nn.Sequential(ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#                                 # nn.Conv1d(32, 64, 3, 1, 1),
#                                 nn.BatchNorm1d(64),
#                                 nn.ReLU(),
#                                 nn.MaxPool1d(2, 2)
#                                 )
#         self.p1_3 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#                                 # nn.Conv1d(64, 64, 3, 1, 1),
#                                 nn.BatchNorm1d(64),
#                                 nn.ReLU(),
#                                 nn.MaxPool1d(2, 2)
#                                 )
#         # 阈值降噪
#         self.p2_1 = sage(channel=64, gap_size=1)  # , 64, 1551
#         self.p2_2 = ECA_block(64)
#         self.p2_3 = nn.Sequential(
#                                   ConvQuadraticOperation(64, 64, 43, 4, 18),
#                                   # nn.Conv1d(64, 64, 43, 4, bias=True),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU()
#                                   )         # , 32, 378
#         self.p2_4 = nn.MaxPool1d(2)         # , 32, 189
#
#         self.p3_1 = nn.Sequential(
#                                   CausalDilationConv1d(16, 32, kernel_size=3, stride=1, dilation=5),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=1),
#                                   nn.BatchNorm1d(32),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p3_2 = nn.Sequential(
#                                   # ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#                                   nn.Conv1d(32, 64, 3, 1, padding=1),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU()
#                                   )
#         # self.p3_3 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#         #                           # nn.Conv1d(32, 32, 3, 4, dilation=3),
#         #                           nn.BatchNorm1d(64),
#         #                           nn.Dropout(0.1),
#         #                           nn.LeakyReLU()
#         #                           )
#         self.p3_4 = nn.MaxPool1d(2)
#         self.p4_1 = nn.Sequential(CausalDilationConv1d(16, 32, kernel_size=3, stride=1, dilation=3),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=3),
#                                   nn.BatchNorm1d(32),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p4_2 = nn.Sequential(
#                                   # ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#                                   nn.Conv1d(32, 64, 3, 1, padding=1),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   )
#         # self.p4_3 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#         #                           # nn.Conv1d(32, 32, 3, 4, dilation=3),
#         #                           nn.BatchNorm1d(64),
#         #                           nn.Dropout(0.1),
#         #                           nn.LeakyReLU()
#         #                           )
#         self.p4_4 = nn.MaxPool1d(2)
#         self.p5_1 = nn.Sequential(CausalDilationConv1d(16, 32, kernel_size=3, stride=1, dilation=1),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=3),
#                                   nn.BatchNorm1d(32),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p5_2 = nn.Sequential(
#                                   # ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#                                   nn.Conv1d(32, 64, 3, 1, padding=1),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   )
#         # self.p5_3 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#         #                           # nn.Conv1d(32, 32, 3, 4, dilation=3),
#         #                           nn.BatchNorm1d(64),
#         #                           nn.Dropout(0.1),
#         #                           nn.LeakyReLU()
#         #                           )
#         self.p5_4 = nn.MaxPool1d(2)
#         # self.pout = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#         #                           # nn.Conv1d(32, 32, 3, 4, dilation=3),
#         #                           nn.BatchNorm1d(64),
#         #                           nn.Dropout(0.1),
#         #                           nn.LeakyReLU()
#         #                           )
#         # self.enc = ECA_block(64)
#         self.fc1 = nn.Linear(12352, 100)
#         self.relu1 = nn.ReLU()
#         self.dp = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(100, 10)
#
#     def forward(self, x):
#         x = self.p1(x)
#         # p3 = self.p3_4(self.p3_3(self.p3_2(self.p3_1(x))))
#         # p4 = self.p4_4(self.p4_3(self.p4_2(self.p4_1(x))))
#         # p5 = self.p5_4(self.p5_3(self.p5_2(self.p5_1(x))))
#         p3 = self.p3_4(self.p3_2(self.p3_1(x)))
#         p4 = self.p4_4(self.p4_2(self.p4_1(x)))
#         p5 = self.p5_4(self.p5_2(self.p5_1(x)))
#         p1 = self.p1_1(x)
#         p1 = self.p1_2(p1)
#         p1 = self.p1_3(p1)
#         p2 = self.p2_4(self.p2_3(self.p2_2(p1 + self.p2_1(p1))))
#
#         out1 = torch.cat((p3, p4, p5), dim=2)
#         # out1 = self.pout(out1)
#
#         # out1 = torch.cat((p1out, torch.cat((p5, torch.cat((p3, p4), dim=2)), dim=2)), dim=2)
#         # Upsample features p3, p4, and p5 to have the same size as p2
#         out1_upsampled = F.interpolate(out1, scale_factor=2, mode='linear', align_corners=False)
#
#         out1 = torch.cat((out1_upsampled, p2), dim=2)
#
#         out = self.fc1(out1.view(x.size(0), -1))
#         out = self.relu1(out)
#         out = self.dp(out)
#         out = self.fc2(out)
#         return out

# class mymodel1_4(nn.Module):
#     """
#     QCNN builder
#     """
#
#     def __init__(self, ) -> object:
#         super(mymodel1_4, self).__init__()
#         self.p1 = nn.Sequential(
#                                 nn.Conv1d(1, 16, 3, 1, 1),
#                                 ConvQuadraticOperation(1, 16, kernel_size=16, stride=3, padding=1),
#                                 nn.BatchNorm1d(16),
#                                 nn.ReLU(),
#                                 nn.MaxPool1d(2, 2)
#                                 )
#         self.p2 = nn.Sequential(
#                                 nn.Conv1d(16, 32, 3, 1, 1),
#                                 ConvQuadraticOperation(16, 32, kernel_size=3, stride=1, padding=1),
#                                 nn.BatchNorm1d(32),
#                                 nn.ReLU(),
#                                 nn.MaxPool1d(2, 2)
#                                 )
#         self.p3 = nn.Sequential(
#                                 nn.Conv1d(32, 64, 3, 1, 1),
#                                 ConvQuadraticOperation(1, 16, kernel_size=16, stride=3, padding=1),
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
#         self.p6 = nn.Sequential(nn.Conv1d(64, 64, 3, 1, 0),
#                                 nn.BatchNorm1d(64),
#                                 nn.ReLU(),
#                                 nn.MaxPool1d(2, 2)
#                                 )
#
#         self.fc1 = nn.Linear(1984, 100)
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
#         out1 = self.p6(x)
#         out = self.fc1(out1.view(x.size(0), -1))
#         out = self.relu1(out)
#         out = self.dp(out)
#         out = self.fc2(out)
#         return F.softmax(out, dim=1)


# 2.0:  0-3:0.8973  0-2:0.9173  0-1:0.9993     -2:0.3920
# class mymodel1_4(nn.Module):
#     """
#     QCNN builder
#     """
#
#     def __init__(self, ) -> object:
#         super(mymodel1_4, self).__init__()
#         self.p1 = nn.Sequential(ConvQuadraticOperation(1, 16, kernel_size=3, stride=1, padding=1),
#                                 # nn.Conv1d(1, 16, 3, 1, 1),
#                                 nn.BatchNorm1d(16),
#                                 nn.ReLU(),
#                                 nn.MaxPool1d(2, 2)
#                                 )
#
#         self.p1_1 = nn.Sequential(ConvQuadraticOperation(16, 32, kernel_size=3, stride=1, padding=1),
#                                 # nn.Conv1d(16, 32, 3, 1, 1),
#                                 nn.BatchNorm1d(32),
#                                 nn.ReLU(),
#                                 nn.MaxPool1d(2, 2)
#                                 )
#         self.p1_2 = nn.Sequential(ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#                                 # nn.Conv1d(32, 64, 3, 1, 1),
#                                 nn.BatchNorm1d(64),
#                                 nn.ReLU(),
#                                 nn.MaxPool1d(2, 2)
#                                 )
#         self.p1_3 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#                                 # nn.Conv1d(64, 64, 3, 1, 1),
#                                 nn.BatchNorm1d(64),
#                                 nn.ReLU(),
#                                 nn.MaxPool1d(2, 2)
#                                 )
#         # 阈值降噪
#         self.p2_1 = sage(channel=64, gap_size=1)  # , 64, 1551
#         self.p2_2 = ECA_block(64)
#         self.p2_3 = nn.Sequential(ConvQuadraticOperation(64, 64, 43, 4, 18),
#                                   # nn.Conv1d(64, 32, 43, 4, bias=True),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU()
#                                   )         # , 32, 378
#         self.p2_4 = nn.MaxPool1d(2)         # , 32, 189
#
#         self.p3_1 = nn.Sequential(
#                                   CausalDilationConv1d(32, 64, kernel_size=3, stride=1, dilation=2),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=1),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p3_2 = nn.Sequential(
#                                   # ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#                                   nn.Conv1d(64, 64, 3, 1, padding=1),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU()
#                                   )
#         self.p3_3 = nn.Sequential(
#                                   # ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#                                   nn.Conv1d(64, 64, 3, 1, padding=0),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU()
#                                   )
#         self.p3_4 = nn.MaxPool1d(2)
#         self.p4_1 = nn.Sequential(CausalDilationConv1d(32, 64, kernel_size=3, stride=1, dilation=3),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=3),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p4_2 = nn.Sequential(
#                                   # ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#                                   nn.Conv1d(64, 64, 3, 1, padding=1),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   )
#         self.p4_3 = nn.Sequential(
#                                   # ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#                                   nn.Conv1d(64, 64, 3, 1, padding=0),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU()
#                                   )
#         self.p4_4 = nn.MaxPool1d(2)
#         self.p5_1 = nn.Sequential(CausalDilationConv1d(32, 64, kernel_size=3, stride=1, dilation=1),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=3),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p5_2 = nn.Sequential(
#                                   # ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#                                   nn.Conv1d(64, 64, 3, 1, padding=1),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   )
#         self.p5_3 = nn.Sequential(
#                                   # ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#                                   nn.Conv1d(64, 64, 3, 1, padding=0),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU()
#                                   )
#         self.p5_4 = nn.MaxPool1d(2)
#         self.fc1 = nn.Linear(25344, 100)
#         self.relu1 = nn.ReLU()
#         self.dp = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(100, 10)
#
#     def forward(self, x):
#         x = self.p1(x)
#         x = self.p1_1(x)
#         p1 = self.p1_2(x)
#         p1 = self.p1_3(p1)
#         p2 = self.p2_4(self.p2_3(self.p2_2(p1 + self.p2_1(p1))))
#         p3 = self.p3_4(self.p3_3(self.p3_2(self.p3_1(x))))
#         p4 = self.p4_4(self.p4_3(self.p4_2(self.p4_1(x))))
#         p5 = self.p5_4(self.p5_3(self.p5_2(self.p5_1(x))))
#         # out1 = torch.cat((p3, p4, p5), dim=2)
#         # out1 = self.pout(out1)
#
#         # out1 = torch.cat((p1out, torch.cat((p5, torch.cat((p3, p4), dim=2)), dim=2)), dim=2)
#         # Upsample features p3, p4, and p5 to have the same size as p2
#         # out1_upsampled = F.interpolate(out1, scale_factor=2, mode='linear', align_corners=False)
#         #
#         # out1 = torch.cat((out1_upsampled, p2), dim=2)
#         # out1 = torch.cat((out1, torch.cat((p5, torch.cat((p3, p4), dim=2)), dim=2)), dim=2)
#         out1 = torch.cat((p2, p5, p4, p3), dim=2)
#         out = self.fc1(out1.view(x.size(0), -1))
#         out = self.relu1(out)
#         out = self.dp(out)
#         out = self.fc2(out)
#         # return F.softmax(out, dim=1)
#         return out

# class mymodel1_4(nn.Module):
#     """
#     QCNN builder
#     """
#
#     def __init__(self, ) -> object:
#         super(mymodel1_4, self).__init__()
#         self.p1 = nn.Sequential(ConvQuadraticOperation(1, 16, kernel_size=3, stride=1, padding=1),
#                                 # nn.Conv1d(1, 16, 3, 1, 1),
#                                 nn.BatchNorm1d(16),
#                                 nn.ReLU(),
#                                 nn.MaxPool1d(2, 2)
#                                 )
#
#         self.p1_1 = nn.Sequential(ConvQuadraticOperation(16, 32, kernel_size=3, stride=1, padding=1),
#                                 # nn.Conv1d(16, 32, 3, 1, 1),
#                                 nn.BatchNorm1d(32),
#                                 nn.ReLU(),
#                                 nn.MaxPool1d(2, 2)
#                                 )
#         self.p1_2 = nn.Sequential(ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#                                 # nn.Conv1d(32, 64, 3, 1, 1),
#                                 nn.BatchNorm1d(64),
#                                 nn.ReLU(),
#                                 nn.MaxPool1d(2, 2)
#                                 )
#         self.p1_3 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#                                 # nn.Conv1d(64, 64, 3, 1, 1),
#                                 nn.BatchNorm1d(64),
#                                 nn.ReLU(),
#                                 nn.MaxPool1d(2, 2)
#                                 )
#         # 阈值降噪
#         self.p2_1 = sage(channel=64, gap_size=1)  # , 64, 1551
#         self.p2_2 = ECA_block(64)
#         self.p2_3 = nn.Sequential(ConvQuadraticOperation(64, 64, 43, 4, 18),
#                                   # nn.Conv1d(64, 32, 43, 4, bias=True),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU()
#                                   )         # , 32, 378
#         self.p2_4 = nn.MaxPool1d(2)         # , 32, 189
#
#         self.p3_1 = nn.Sequential(
#                                   CausalDilationConv1d(32, 64, kernel_size=3, stride=1, dilation=2),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=1),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p3_2 = nn.Sequential(
#                                   # ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#                                   nn.Conv1d(64, 64, 3, 1, padding=1),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU()
#                                   )
#         self.p3_4 = nn.MaxPool1d(2)
#         self.p4_1 = nn.Sequential(CausalDilationConv1d(32, 64, kernel_size=3, stride=1, dilation=3),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=3),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p4_2 = nn.Sequential(
#                                   # ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#                                   nn.Conv1d(64, 64, 3, 1, padding=1),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   )
#         self.p4_4 = nn.MaxPool1d(2)
#         self.p5_1 = nn.Sequential(CausalDilationConv1d(32, 64, kernel_size=3, stride=1, dilation=1),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=3),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p5_2 = nn.Sequential(
#                                   # ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#                                   nn.Conv1d(64, 64, 3, 1, padding=1),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   )
#         self.p5_4 = nn.MaxPool1d(2)
#         self.fc1 = nn.Linear(28672, 100)
#         self.relu1 = nn.ReLU()
#         self.dp = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(100, 10)
#
#     def forward(self, x):
#         x = self.p1(x)
#         x = self.p1_1(x)
#         p1 = self.p1_2(x)
#         p1 = self.p1_3(p1)
#         p2 = self.p2_4(self.p2_2(p1 + self.p2_1(p1)))
#         p3 = self.p3_4(self.p3_2(self.p3_1(x)))
#         p4 = self.p4_4(self.p4_2(self.p4_1(x)))
#         p5 = self.p5_4(self.p5_2(self.p5_1(x)))
#         # out1 = torch.cat((p3, p4, p5), dim=2)
#         # out1 = self.pout(out1)
#
#         # out1 = torch.cat((p1out, torch.cat((p5, torch.cat((p3, p4), dim=2)), dim=2)), dim=2)
#         # Upsample features p3, p4, and p5 to have the same size as p2
#         # out1_upsampled = F.interpolate(out1, scale_factor=2, mode='linear', align_corners=False)
#         #
#         # out1 = torch.cat((out1_upsampled, p2), dim=2)
#         # out1 = torch.cat((out1, torch.cat((p5, torch.cat((p3, p4), dim=2)), dim=2)), dim=2)
#         out1 = torch.cat((p2, p5, p4, p3), dim=2)
#         out = self.fc1(out1.view(x.size(0), -1))
#         out = self.relu1(out)
#         out = self.dp(out)
#         out = self.fc2(out)
#         # return F.softmax(out, dim=1)
#         return out
from Model.lapeplace import Laplace_fastv2 as fast
# 0. 9027  0.9023357787995698
# class mymodel1_4(nn.Module):
#     """
#     QCNN builder
#     """
#
#     def __init__(self, ) -> object:
#         super(mymodel1_4, self).__init__()
#         self.p1 = nn.Sequential(ConvQuadraticOperation(1, 16, kernel_size=64, stride=8, padding=28),
#                                 # nn.Conv1d(1, 16, 3, 1, 1),
#                                 nn.BatchNorm1d(16),
#                                 nn.ReLU(),
#                                 nn.MaxPool1d(2, 2)
#                                 )
#
#         self.p1_1 = nn.Sequential(ConvQuadraticOperation(16, 32, kernel_size=3, stride=1, padding=1),
#                                 # nn.Conv1d(16, 32, 3, 1, 1),
#                                 nn.BatchNorm1d(32),
#                                 nn.ReLU(),
#                                 nn.MaxPool1d(2, 2)
#                                 )
#         self.p1_2 = nn.Sequential(ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#                                 # nn.Conv1d(32, 64, 3, 1, 1),
#                                 nn.BatchNorm1d(64),
#                                 nn.ReLU(),
#                                 # nn.MaxPool1d(2, 2)
#                                 )
#         # self.p1_3 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#         #                         # nn.Conv1d(64, 64, 3, 1, 1),
#         #                         nn.BatchNorm1d(64),
#         #                         nn.ReLU(),
#         #                         nn.MaxPool1d(2, 2)
#         #                         )
#         # self.p1_4 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#         #                           # nn.Conv1d(64, 64, 3, 1, 1),
#         #                           nn.BatchNorm1d(64),
#         #                           nn.ReLU(),
#         #                           nn.MaxPool1d(2, 2)
#         #                           )
#         # self.p1_5 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=0),
#         #                           # nn.Conv1d(64, 64, 3, 1, 1),
#         #                           nn.BatchNorm1d(64),
#         #                           nn.ReLU(),
#         #                           nn.MaxPool1d(2, 2)
#         #                           )
#         # 阈值降噪
#         self.p2_1 = sage(channel=64, gap_size=1)  # , 64, 1551
#         self.p2_2 = ECA_block(64)
#         # self.p2_3 = nn.Sequential(ConvQuadraticOperation(64, 64, 43, 4, 18),
#         #                           # nn.Conv1d(64, 32, 43, 4, bias=True),
#         #                           nn.BatchNorm1d(64),
#         #                           nn.Dropout(0.1),
#         #                           nn.LeakyReLU()
#         #                           )         # , 32, 378
#         self.p2_3 = nn.Sequential(ConvQuadraticOperation(64, 64, 32, 4, 48),
#                                   # nn.Conv1d(64, 32, 43, 4, bias=True),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   # nn.ReLU(),
#                                   )  # , 32, 378
#         self.p2_4 = nn.MaxPool1d(2)         # , 32, 189
#         #
#         self.p3_1 = nn.Sequential(
#                                   CausalDilationConv1d(16, 32, kernel_size=3, stride=1, dilation=5),
#                                   # nn.Conv1d(16, 32, 3, 1, dilation=5, padding=1),
#                                   nn.BatchNorm1d(32),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   # nn.ReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p3_2 = nn.Sequential(
#                                   ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=3),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   # nn.ReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p3_3 = nn.Sequential(
#                                   ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=3),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   # nn.ReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#
#         self.p4_1 = nn.Sequential(
#                                   CausalDilationConv1d(16, 32, kernel_size=3, stride=1, dilation=3),
#                                   # nn.Conv1d(16, 32, 3, 1, dilation=3, padding=1),
#                                   nn.BatchNorm1d(32),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   # nn.ReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p4_2 = nn.Sequential(
#                                   ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=3),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   # nn.ReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p4_3 = nn.Sequential(
#                                   ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=3),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   # nn.ReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#
#         self.p5_1 = nn.Sequential(
#                                   CausalDilationConv1d(16, 32, kernel_size=3, stride=1, dilation=1),
#                                   # nn.Conv1d(16, 32, 3, 1, dilation=1, padding=0),
#                                   nn.BatchNorm1d(32),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   # nn.ReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p5_2 = nn.Sequential(
#                                   ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=3),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   # nn.ReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p5_3 = nn.Sequential(
#                                   ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=3),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   # nn.ReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         # self
#         # self.gru = nn.GRU(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
#         self.fc1 = nn.Linear(1024, 100)
#         self.eca = ECA_block(100)
#         # self.relu1 = nn.LeakyReLU()         # 8893
#         self.relu1 = nn.ReLU()
#         self.dp = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(100, 10)
#     #     self._initialize_weights()
#     #
#     # def _initialize_weights(self):
#     #     for m in self.modules():
#     #         if isinstance(m, nn.Conv1d):
#     #             nn.init.kaiming_normal_(m.weight.data)
#     #             nn.init.constant_(m.bias.data, 0.0)
#     #         elif isinstance(m, nn.BatchNorm1d):
#     #             m.weight.data.fill_(1)
#     #             m.bias.data.fill_(0)
#     #         elif isinstance(m, nn.Linear):
#     #             m.weight.data.normal_(0, 0.01)
#     #             if m.bias is not None:
#     #                 m.bias.data.fill_(1)
#
#     def forward(self, x):
#         x = self.p1(x)
#         # p3 = self.p3_3(self.p3_2(self.p3_1(x)))
#         p4 = self.p4_3(self.p4_2(self.p4_1(x)))
#         p5 = self.p5_3(self.p5_2(self.p5_1(x)))
#         p1 = self.p1_1(x)
#         p1 = self.p1_2(p1)
#         # p1 = self.p1_3(p1)
#         # p1 = self.p1_4(p1)
#         # p1 = self.p1_5(p1)
#         p1out = self.p2_4(self.p2_3(self.p2_2(p1 + self.p2_1(p1))))
#         # p1out = self.p2_4(self.p2_3(p1 + self.p2_1(p1)))
#         # out1 = torch.cat((p1out, torch.cat((p5, torch.cat((p3, p4), dim=2)), dim=2)), dim=2)
#         # out1 = torch.cat((p1out, torch.cat((p4, p5), dim=2)), dim=2)
#         # out = torch.add(p1out, torch.add(p5, torch.add(p4, p3)))
#         out = torch.add(p1out, torch.add(p5, p4))
#         # out = self.eca(out)
#         out = self.fc1(out.view(x.size(0), -1))
#         # out = self.fc1(out)
#         out = self.relu1(out)
#         out = self.dp(out)
#         out = self.fc2(out)
#         # return F.softmax(out, dim=1)
#         return out




###就决定是你了！
# 到100epoch才刚开始收敛，准确率0.9240 0.9229399014667823 变工况的话都在80左右 不加噪收敛倒是挺快的 图也挺好的
# -2:0.9327(0.9320278675134219)
# class mymodel1_4(nn.Module):
#     """
#     QCNN builder
#     """
#
#     def __init__(self, ) -> object:
#         super(mymodel1_4, self).__init__()
#         self.p1 = nn.Sequential(ConvQuadraticOperation(1, 16, kernel_size=64, stride=8, padding=28),
#                                 # nn.Conv1d(1, 16, 3, 1, 1),
#                                 nn.BatchNorm1d(16),
#                                 nn.Dropout(0.1),
#                                 nn.LeakyReLU(),
#                                 nn.MaxPool1d(2, 2)
#                                 )
#
#         self.p1_1 = nn.Sequential(ConvQuadraticOperation(16, 32, kernel_size=3, stride=1, padding=1),
#                                 # nn.Conv1d(16, 32, 3, 1, 1),
#                                 nn.BatchNorm1d(32),
#                                 nn.Dropout(0.1),
#                                 nn.LeakyReLU(),
#                                 nn.MaxPool1d(2, 2)
#                                 )
#         self.p1_2 = nn.Sequential(ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#                                 # nn.Conv1d(32, 64, 3, 1, 1),
#                                 nn.BatchNorm1d(64),
#                                 nn.Dropout(0.1),
#                                 nn.LeakyReLU(),
#                                 # nn.MaxPool1d(2, 2)
#                                 )
#         # self.p1_3 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#         #                         # nn.Conv1d(64, 64, 3, 1, 1),
#         #                         nn.BatchNorm1d(64),
#         #                         nn.ReLU(),
#         #                         nn.MaxPool1d(2, 2)
#         #                         )
#         # self.p1_4 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#         #                           # nn.Conv1d(64, 64, 3, 1, 1),
#         #                           nn.BatchNorm1d(64),
#         #                           nn.ReLU(),
#         #                           nn.MaxPool1d(2, 2)
#         #                           )
#         # self.p1_5 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=0),
#         #                           # nn.Conv1d(64, 64, 3, 1, 1),
#         #                           nn.BatchNorm1d(64),
#         #                           nn.ReLU(),
#         #                           nn.MaxPool1d(2, 2)
#         #                           )
#         # 阈值降噪
#         self.p2_1 = sage(channel=64, gap_size=1)  # , 64, 1551
#         self.p2_2 = ECA_block(64)
#         # self.p2_3 = nn.Sequential(ConvQuadraticOperation(64, 64, 43, 4, 18),
#         #                           # nn.Conv1d(64, 32, 43, 4, bias=True),
#         #                           nn.BatchNorm1d(64),
#         #                           nn.Dropout(0.1),
#         #                           nn.LeakyReLU()
#         #                           )         # , 32, 378
#         self.p2_3 = nn.Sequential(ConvQuadraticOperation(64, 64, 32, 4, 48),
#         # self.p2_3 = nn.Sequential(ConvQuadraticOperation(64, 64, 32, 4, 32),
#                                   # nn.Conv1d(64, 32, 43, 4, bias=True),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU()
#                                   )  # , 32, 378
#         self.p2_4 = nn.MaxPool1d(2)         # , 32, 189
#
#         self.p3_1 = nn.Sequential(
#                                   CausalDilationConv1d(16, 32, kernel_size=3, stride=1, dilation=5),
#                                   # nn.Conv1d(16, 32, 3, 1, dilation=5),
#                                   nn.BatchNorm1d(32),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p3_2 = nn.Sequential(
#                                   ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#                                   # nn.Conv1d(32, 64, 3, 1, padding=1),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p3_3 = nn.Sequential(
#                                   ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#                                   # nn.Conv1d(64, 64, 3, 1, padding=1),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   ECA_block(64),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#
#         self.p4_1 = nn.Sequential(
#                                   CausalDilationConv1d(16, 32, kernel_size=3, stride=1, dilation=3),
#                                   # nn.Conv1d(16, 32, 3, 1, dilation=3),
#                                   nn.BatchNorm1d(32),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p4_2 = nn.Sequential(
#                                   ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#                                   # nn.Conv1d(32, 64, 3, 1, padding=1),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p4_3 = nn.Sequential(
#                                   ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#                                   # nn.Conv1d(64, 64, 3, 1, padding=1),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   ECA_block(64),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#
#         self.p5_1 = nn.Sequential(
#                                   CausalDilationConv1d(16, 32, kernel_size=3, stride=1, dilation=1),
#                                   # nn.Conv1d(16, 32, 3, 1, dilation=1),
#                                   nn.BatchNorm1d(32),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p5_2 = nn.Sequential(
#                                   ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#                                   # nn.Conv1d(32, 64, 3, 1, padding=1),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p5_3 = nn.Sequential(
#                                   ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#                                   # nn.Conv1d(64, 64, 3, 1, padding=1),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   ECA_block(64),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         # self
#         self.fc1 = nn.Linear(1024, 100)
#         # self.fc1 = nn.Linear(512, 100)
#         self.eca = ECA_block(100)
#         self.relu1 = nn.LeakyReLU()
#         self.dp = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(100, 10)
#     #     self._initialize_weights()
#     #
#     # def _initialize_weights(self):
#     #     for m in self.modules():
#     #         if isinstance(m, nn.Conv1d):
#     #             nn.init.kaiming_normal_(m.weight.data)
#     #             nn.init.constant_(m.bias.data, 0.0)
#     #         elif isinstance(m, nn.BatchNorm1d):
#     #             m.weight.data.fill_(1)
#     #             m.bias.data.fill_(0)
#     #         elif isinstance(m, nn.Linear):
#     #             m.weight.data.normal_(0, 0.01)
#     #             if m.bias is not None:
#     #                 m.bias.data.fill_(1)
#
#     def forward(self, x):
#         x = self.p1(x)
#         p3 = self.p3_3(self.p3_2(self.p3_1(x)))
#         p4 = self.p4_3(self.p4_2(self.p4_1(x)))
#         p5 = self.p5_3(self.p5_2(self.p5_1(x)))
#         # p3 = self.p3_1(x)
#         # p4 = self.p4_1(x)
#         # p5 = self.p5_1(x)
#         p1 = self.p1_1(x)
#         p1 = self.p1_2(p1)
#         p1out = self.p2_4(self.p2_3(self.p2_2(p1 + self.p2_1(p1))))
#         # p1out = self.p2_4(self.p2_3(p1 + self.p2_1(p1)))
#         # pout = torch.add(p5, torch.add(p4, p3))
#         # pout = self.p3_3(self.p3_2(pout))
#         out = torch.add(p1out, torch.add(p5, torch.add(p4, p3)))
#         # out = torch.add(p5, torch.add(p4, p3))
#         # out = torch.add(pout, p1out)
#         out = self.eca(out)
#         out = self.fc1(out.view(x.size(0), -1))
#         out = self.relu1(out)
#         out = self.dp(out)
#         out = self.fc2(out)
#         return out

# 训练集没收敛到100只到98 准确率0.9113 0.9107745345774066
# class mymodel1_4(nn.Module):
#     """
#     QCNN builder
#     """
#
#     def __init__(self, ) -> object:
#         super(mymodel1_4, self).__init__()
#         self.p1 = nn.Sequential(ConvQuadraticOperation(1, 16, kernel_size=64, stride=8, padding=28),
#                                 # nn.Conv1d(1, 16, 3, 1, 1),
#                                 nn.BatchNorm1d(16),
#                                 nn.Dropout(0.1),
#                                 nn.LeakyReLU(),
#                                 nn.MaxPool1d(2, 2)
#                                 )
#
#         self.p1_1 = nn.Sequential(ConvQuadraticOperation(16, 32, kernel_size=3, stride=1, padding=1),
#                                 # nn.Conv1d(16, 32, 3, 1, 1),
#                                 nn.BatchNorm1d(32),
#                                 nn.Dropout(0.1),
#                                 nn.LeakyReLU(),
#                                 nn.MaxPool1d(2, 2)
#                                 )
#         self.p1_2 = nn.Sequential(ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#                                 # nn.Conv1d(32, 64, 3, 1, 1),
#                                 nn.BatchNorm1d(64),
#                                 nn.Dropout(0.1),
#                                 nn.LeakyReLU(),
#                                 # nn.MaxPool1d(2, 2)
#                                 )
#         # self.p1_3 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#         #                         # nn.Conv1d(64, 64, 3, 1, 1),
#         #                         nn.BatchNorm1d(64),
#         #                         nn.ReLU(),
#         #                         nn.MaxPool1d(2, 2)
#         #                         )
#         # self.p1_4 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#         #                           # nn.Conv1d(64, 64, 3, 1, 1),
#         #                           nn.BatchNorm1d(64),
#         #                           nn.ReLU(),
#         #                           nn.MaxPool1d(2, 2)
#         #                           )
#         # self.p1_5 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=0),
#         #                           # nn.Conv1d(64, 64, 3, 1, 1),
#         #                           nn.BatchNorm1d(64),
#         #                           nn.ReLU(),
#         #                           nn.MaxPool1d(2, 2)
#         #                           )
#         # 阈值降噪
#         self.p2_1 = sage(channel=64, gap_size=1)  # , 64, 1551
#         self.p2_2 = ECA_block(64)
#         # self.p2_3 = nn.Sequential(ConvQuadraticOperation(64, 64, 43, 4, 18),
#         #                           # nn.Conv1d(64, 32, 43, 4, bias=True),
#         #                           nn.BatchNorm1d(64),
#         #                           nn.Dropout(0.1),
#         #                           nn.LeakyReLU()
#         #                           )         # , 32, 378
#         self.p2_3 = nn.Sequential(ConvQuadraticOperation(64, 64, 32, 4, 48),
#                                   # nn.Conv1d(64, 32, 43, 4, bias=True),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU()
#                                   )  # , 32, 378
#         self.p2_4 = nn.MaxPool1d(2)         # , 32, 189
#         #
#         self.p3_1 = nn.Sequential(CausalDilationConv1d(16, 32, kernel_size=3, stride=1, dilation=5),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=1),
#                                   nn.BatchNorm1d(32),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p3_2 = nn.Sequential(ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=3),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p3_3 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=3),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#
#         self.p4_1 = nn.Sequential(CausalDilationConv1d(16, 32, kernel_size=3, stride=1, dilation=3),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=3),
#                                   nn.BatchNorm1d(32),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p4_2 = nn.Sequential(ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=3),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p4_3 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=3),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#
#         self.p5_1 = nn.Sequential(CausalDilationConv1d(16, 32, kernel_size=3, stride=1, dilation=1),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=3),
#                                   nn.BatchNorm1d(32),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p5_2 = nn.Sequential(ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=3),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p5_3 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=3),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2)
#                                   )
#         # self
#         self.fc1 = nn.Linear(1024, 100)
#         # self.eca = ECA_block(100)
#         self.relu1 = nn.LeakyReLU()
#         self.dp = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(100, 10)
#     #     self._initialize_weights()
#     #
#     # def _initialize_weights(self):
#     #     for m in self.modules():
#     #         if isinstance(m, nn.Conv1d):
#     #             nn.init.kaiming_normal_(m.weight.data)
#     #             nn.init.constant_(m.bias.data, 0.0)
#     #         elif isinstance(m, nn.BatchNorm1d):
#     #             m.weight.data.fill_(1)
#     #             m.bias.data.fill_(0)
#     #         elif isinstance(m, nn.Linear):
#     #             m.weight.data.normal_(0, 0.01)
#     #             if m.bias is not None:
#     #                 m.bias.data.fill_(1)
#
#     def forward(self, x):
#         x = self.p1(x)
#         p3 = self.p3_3(self.p3_2(self.p3_1(x)))
#         p4 = self.p4_3(self.p4_2(self.p4_1(x)))
#         p5 = self.p5_3(self.p5_2(self.p5_1(x)))
#         p1 = self.p1_1(x)
#         p1 = self.p1_2(p1)
#         # p1 = self.p1_3(p1)
#         # p1 = self.p1_4(p1)
#         # p1 = self.p1_5(p1)
#         p1out = self.p2_4(self.p2_3(self.p2_2(p1 + self.p2_1(p1))))
#         # p1out = self.p2_4(self.p2_3(p1 + self.p2_1(p1)))
#         # out1 = torch.cat((p1out, torch.cat((p5, torch.cat((p3, p4), dim=2)), dim=2)), dim=2)
#         # out1 = torch.cat((p1out, torch.cat((p4, p5), dim=2)), dim=2)
#         out = torch.add(p1out, torch.add(p5, torch.add(p4, p3)))
#         # out = self.eca(out)
#         out = self.fc1(out.view(x.size(0), -1))
#         # out = self.fc1(out)
#         out = self.relu1(out)
#         out = self.dp(out)
#         out = self.fc2(out)
#         # return F.softmax(out, dim=1)
#         return out

# 0.9147 0.9136764630921658 最后dropout改成了0.1
# class mymodel1_4(nn.Module):
#     """
#     QCNN builder
#     """
#
#     def __init__(self, ) -> object:
#         super(mymodel1_4, self).__init__()
#         self.p1 = nn.Sequential(ConvQuadraticOperation(1, 16, kernel_size=64, stride=8, padding=28),
#                                 # nn.Conv1d(1, 16, 3, 1, 1),
#                                 nn.BatchNorm1d(16),
#                                 nn.Dropout(0.1),
#                                 nn.LeakyReLU(),
#                                 nn.MaxPool1d(2, 2)
#                                 )
#
#         self.p1_1 = nn.Sequential(ConvQuadraticOperation(16, 32, kernel_size=3, stride=1, padding=1),
#                                 # nn.Conv1d(16, 32, 3, 1, 1),
#                                 nn.BatchNorm1d(32),
#                                 nn.Dropout(0.1),
#                                 nn.LeakyReLU(),
#                                 nn.MaxPool1d(2, 2)
#                                 )
#         self.p1_2 = nn.Sequential(ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#                                 # nn.Conv1d(32, 64, 3, 1, 1),
#                                 nn.BatchNorm1d(64),
#                                 nn.Dropout(0.1),
#                                 nn.LeakyReLU(),
#                                 # nn.MaxPool1d(2, 2)
#                                 )
#         # self.p1_3 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#         #                         # nn.Conv1d(64, 64, 3, 1, 1),
#         #                         nn.BatchNorm1d(64),
#         #                         nn.ReLU(),
#         #                         nn.MaxPool1d(2, 2)
#         #                         )
#         # self.p1_4 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#         #                           # nn.Conv1d(64, 64, 3, 1, 1),
#         #                           nn.BatchNorm1d(64),
#         #                           nn.ReLU(),
#         #                           nn.MaxPool1d(2, 2)
#         #                           )
#         # self.p1_5 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=0),
#         #                           # nn.Conv1d(64, 64, 3, 1, 1),
#         #                           nn.BatchNorm1d(64),
#         #                           nn.ReLU(),
#         #                           nn.MaxPool1d(2, 2)
#         #                           )
#         # 阈值降噪
#         self.p2_1 = sage(channel=64, gap_size=1)  # , 64, 1551
#         self.p2_2 = ECA_block(64)
#         # self.p2_3 = nn.Sequential(ConvQuadraticOperation(64, 64, 43, 4, 18),
#         #                           # nn.Conv1d(64, 32, 43, 4, bias=True),
#         #                           nn.BatchNorm1d(64),
#         #                           nn.Dropout(0.1),
#         #                           nn.LeakyReLU()
#         #                           )         # , 32, 378
#         self.p2_3 = nn.Sequential(ConvQuadraticOperation(64, 64, 32, 4, 48),
#                                   # nn.Conv1d(64, 32, 43, 4, bias=True),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU()
#                                   )  # , 32, 378
#         self.p2_4 = nn.MaxPool1d(2)         # , 32, 189
#         #
#         self.p3_1 = nn.Sequential(CausalDilationConv1d(16, 32, kernel_size=3, stride=1, dilation=5),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=1),
#                                   nn.BatchNorm1d(32),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p3_2 = nn.Sequential(ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=3),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p3_3 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=3),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#
#         self.p4_1 = nn.Sequential(CausalDilationConv1d(16, 32, kernel_size=3, stride=1, dilation=3),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=3),
#                                   nn.BatchNorm1d(32),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p4_2 = nn.Sequential(ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=3),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p4_3 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=3),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#
#         self.p5_1 = nn.Sequential(CausalDilationConv1d(16, 32, kernel_size=3, stride=1, dilation=1),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=3),
#                                   nn.BatchNorm1d(32),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p5_2 = nn.Sequential(ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=3),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p5_3 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=3),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2)
#                                   )
#         # self
#         self.fc1 = nn.Linear(1024, 100)
#         self.eca = ECA_block(100)
#         self.relu1 = nn.LeakyReLU()
#         self.dp = nn.Dropout(0.1)
#         self.fc2 = nn.Linear(100, 10)
#     #     self._initialize_weights()
#     #
#     # def _initialize_weights(self):
#     #     for m in self.modules():
#     #         if isinstance(m, nn.Conv1d):
#     #             nn.init.kaiming_normal_(m.weight.data)
#     #             nn.init.constant_(m.bias.data, 0.0)
#     #         elif isinstance(m, nn.BatchNorm1d):
#     #             m.weight.data.fill_(1)
#     #             m.bias.data.fill_(0)
#     #         elif isinstance(m, nn.Linear):
#     #             m.weight.data.normal_(0, 0.01)
#     #             if m.bias is not None:
#     #                 m.bias.data.fill_(1)
#
#     def forward(self, x):
#         x = self.p1(x)
#         p3 = self.p3_3(self.p3_2(self.p3_1(x)))
#         p4 = self.p4_3(self.p4_2(self.p4_1(x)))
#         p5 = self.p5_3(self.p5_2(self.p5_1(x)))
#         p1 = self.p1_1(x)
#         p1 = self.p1_2(p1)
#         # p1 = self.p1_3(p1)
#         # p1 = self.p1_4(p1)
#         # p1 = self.p1_5(p1)
#         p1out = self.p2_4(self.p2_3(self.p2_2(p1 + self.p2_1(p1))))
#         # p1out = self.p2_4(self.p2_3(p1 + self.p2_1(p1)))
#         # out1 = torch.cat((p1out, torch.cat((p5, torch.cat((p3, p4), dim=2)), dim=2)), dim=2)
#         # out1 = torch.cat((p1out, torch.cat((p4, p5), dim=2)), dim=2)
#         out = torch.add(p1out, torch.add(p5, torch.add(p4, p3)))
#         out = self.eca(out)
#         out = self.fc1(out.view(x.size(0), -1))
#         # out = self.fc1(out)
#         out = self.relu1(out)
#         out = self.dp(out)
#         out = self.fc2(out)
#         # return F.softmax(out, dim=1)
#         return out


# 0.9180 但没收敛只到95
# class mymodel1_4(nn.Module):
#     """
#     QCNN builder
#     """
#
#     def __init__(self, ) -> object:
#         super(mymodel1_4, self).__init__()
#         self.p1 = nn.Sequential(ConvQuadraticOperation(1, 16, kernel_size=64, stride=8, padding=28),
#                                 # nn.Conv1d(1, 16, 3, 1, 1),
#                                 nn.BatchNorm1d(16),
#                                 nn.Dropout(0.1),
#                                 nn.LeakyReLU(),
#                                 nn.MaxPool1d(2, 2)
#                                 )
#
#         self.p1_1 = nn.Sequential(ConvQuadraticOperation(16, 32, kernel_size=3, stride=1, padding=28),
#                                 # nn.Conv1d(16, 32, 3, 1, 1),
#                                 nn.BatchNorm1d(32),
#                                 nn.Dropout(0.1),
#                                 nn.LeakyReLU(),
#                                 # nn.MaxPool1d(2, 2)
#                                 )
#         self.p1_2 = nn.Sequential(ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=6),
#                                 # nn.Conv1d(32, 64, 3, 1, 1),
#                                 nn.BatchNorm1d(64),
#                                 nn.Dropout(0.1),
#                                 nn.LeakyReLU(),
#                                 # nn.MaxPool1d(2, 2)
#                                 )
#         # self.p1_3 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#         #                         # nn.Conv1d(64, 64, 3, 1, 1),
#         #                         nn.BatchNorm1d(64),
#         #                         nn.ReLU(),
#         #                         nn.MaxPool1d(2, 2)
#         #                         )
#         # self.p1_4 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#         #                           # nn.Conv1d(64, 64, 3, 1, 1),
#         #                           nn.BatchNorm1d(64),
#         #                           nn.ReLU(),
#         #                           nn.MaxPool1d(2, 2)
#         #                           )
#         # self.p1_5 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=0),
#         #                           # nn.Conv1d(64, 64, 3, 1, 1),
#         #                           nn.BatchNorm1d(64),
#         #                           nn.ReLU(),
#         #                           nn.MaxPool1d(2, 2)
#         #                           )
#         # 阈值降噪
#         self.p2_1 = sage(channel=64, gap_size=1)  # , 64, 1551
#         self.p2_2 = ECA_block(64)
#         # self.p2_3 = nn.Sequential(ConvQuadraticOperation(64, 64, 43, 4, 18),
#         #                           # nn.Conv1d(64, 32, 43, 4, bias=True),
#         #                           nn.BatchNorm1d(64),
#         #                           nn.Dropout(0.1),
#         #                           nn.LeakyReLU()
#         #                           )         # , 32, 378
#         self.p2_3 = nn.Sequential(ConvQuadraticOperation(64, 64, 32, 4, 48),
#                                   # nn.Conv1d(64, 32, 43, 4, bias=True),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU()
#                                   )  # , 32, 378
#         self.p2_4 = nn.MaxPool1d(2)         # , 32, 189
#         #
#         self.p3_1 = nn.Sequential(CausalDilationConv1d(16, 32, kernel_size=3, stride=1, dilation=5),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=1),
#                                   nn.BatchNorm1d(32),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p3_2 = nn.Sequential(ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=3),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         # self.p3_3 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#         #                           # nn.Conv1d(32, 32, 3, 4, dilation=3),
#         #                           nn.BatchNorm1d(64),
#         #                           nn.Dropout(0.1),
#         #                           nn.LeakyReLU(),
#         #                           nn.MaxPool1d(2, 2)
#         #                           )
#
#         self.p4_1 = nn.Sequential(CausalDilationConv1d(16, 32, kernel_size=3, stride=1, dilation=3),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=3),
#                                   nn.BatchNorm1d(32),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p4_2 = nn.Sequential(ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=3),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         # self.p4_3 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#         #                           # nn.Conv1d(32, 32, 3, 4, dilation=3),
#         #                           nn.BatchNorm1d(64),
#         #                           nn.Dropout(0.1),
#         #                           nn.LeakyReLU(),
#         #                           nn.MaxPool1d(2, 2)
#         #                           )
#
#         self.p5_1 = nn.Sequential(CausalDilationConv1d(16, 32, kernel_size=3, stride=1, dilation=1),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=3),
#                                   nn.BatchNorm1d(32),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p5_2 = nn.Sequential(ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#                                   # nn.Conv1d(32, 32, 3, 4, dilation=3),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         # self.p5_3 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#         #                           # nn.Conv1d(32, 32, 3, 4, dilation=3),
#         #                           nn.BatchNorm1d(64),
#         #                           nn.Dropout(0.1),
#         #                           nn.LeakyReLU(),
#         #                           nn.MaxPool1d(2)
#         #                           )
#         # self
#         self.fc1 = nn.Linear(2048, 100)
#         self.eca = ECA_block(100)
#         self.relu1 = nn.LeakyReLU()
#         self.dp = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(100, 10)
#     #     self._initialize_weights()
#     #
#     # def _initialize_weights(self):
#     #     for m in self.modules():
#     #         if isinstance(m, nn.Conv1d):
#     #             nn.init.kaiming_normal_(m.weight.data)
#     #             nn.init.constant_(m.bias.data, 0.0)
#     #         elif isinstance(m, nn.BatchNorm1d):
#     #             m.weight.data.fill_(1)
#     #             m.bias.data.fill_(0)
#     #         elif isinstance(m, nn.Linear):
#     #             m.weight.data.normal_(0, 0.01)
#     #             if m.bias is not None:
#     #                 m.bias.data.fill_(1)
#
#     def forward(self, x):
#         x = self.p1(x)
#         p3 = self.p3_2(self.p3_1(x))
#         p4 = self.p4_2(self.p4_1(x))
#         p5 = self.p5_2(self.p5_1(x))
#         p1 = self.p1_1(x)
#         p1 = self.p1_2(p1)
#         # p1 = self.p1_3(p1)
#         # p1 = self.p1_4(p1)
#         # p1 = self.p1_5(p1)
#         p1out = self.p2_4(self.p2_3(self.p2_2(p1 + self.p2_1(p1))))
#         # p1out = self.p2_4(self.p2_3(p1 + self.p2_1(p1)))
#         # out1 = torch.cat((p1out, torch.cat((p5, torch.cat((p3, p4), dim=2)), dim=2)), dim=2)
#         # out1 = torch.cat((p1out, torch.cat((p4, p5), dim=2)), dim=2)
#         out = torch.add(p1out, torch.add(p5, torch.add(p4, p3)))
#         out = self.eca(out)
#         out = self.fc1(out.view(x.size(0), -1))
#         # out = self.fc1(out)
#         out = self.relu1(out)
#         out = self.dp(out)
#         out = self.fc2(out)
#         # return F.softmax(out, dim=1)
#         return out


# -2:0.9320(0.9313869709925298)     三条分支换成普通卷积-2:0.9300(0.928787936279932)
# class mymodel1_4(nn.Module):
#     """
#     QCNN builder
#     """
#
#     def __init__(self, ) -> object:
#         super(mymodel1_4, self).__init__()
#         self.p1 = nn.Sequential(ConvQuadraticOperation(1, 16, kernel_size=64, stride=8, padding=28),
#                                 # nn.Conv1d(1, 16, 3, 1, 1),
#                                 nn.BatchNorm1d(16),
#                                 nn.Dropout(0.1),
#                                 nn.LeakyReLU(),
#                                 nn.MaxPool1d(2, 2)
#                                 )
#
#         self.p1_1 = nn.Sequential(ConvQuadraticOperation(16, 32, kernel_size=3, stride=1, padding=1),
#                                 # nn.Conv1d(16, 32, 3, 1, 1),
#                                 nn.BatchNorm1d(32),
#                                 nn.Dropout(0.1),
#                                 nn.LeakyReLU(),
#                                 nn.MaxPool1d(2, 2)
#                                 )
#         self.p1_2 = nn.Sequential(ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#                                 # nn.Conv1d(32, 64, 3, 1, 1),
#                                 nn.BatchNorm1d(64),
#                                 nn.Dropout(0.1),
#                                 nn.LeakyReLU(),
#                                 # nn.MaxPool1d(2, 2)
#                                 )
#         # self.p1_3 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#         #                         # nn.Conv1d(64, 64, 3, 1, 1),
#         #                         nn.BatchNorm1d(64),
#         #                         nn.ReLU(),
#         #                         nn.MaxPool1d(2, 2)
#         #                         )
#         # self.p1_4 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#         #                           # nn.Conv1d(64, 64, 3, 1, 1),
#         #                           nn.BatchNorm1d(64),
#         #                           nn.ReLU(),
#         #                           nn.MaxPool1d(2, 2)
#         #                           )
#         # self.p1_5 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=0),
#         #                           # nn.Conv1d(64, 64, 3, 1, 1),
#         #                           nn.BatchNorm1d(64),
#         #                           nn.ReLU(),
#         #                           nn.MaxPool1d(2, 2)
#         #                           )
#         # 阈值降噪
#         self.p2_1 = sage(channel=64, gap_size=1)  # , 64, 1551
#         self.p2_2 = ECA_block(64)
#         # self.p2_3 = nn.Sequential(ConvQuadraticOperation(64, 64, 43, 4, 18),
#         #                           # nn.Conv1d(64, 32, 43, 4, bias=True),
#         #                           nn.BatchNorm1d(64),
#         #                           nn.Dropout(0.1),
#         #                           nn.LeakyReLU()
#         #                           )         # , 32, 378
#         self.p2_3 = nn.Sequential(ConvQuadraticOperation(64, 64, 32, 4, 44),
#         # self.p2_3 = nn.Sequential(ConvQuadraticOperation(64, 64, 32, 4, 32),
#                                   # nn.Conv1d(64, 32, 43, 4, bias=True),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU()
#                                   )  # , 32, 378
#         self.p2_4 = nn.MaxPool1d(2)         # , 32, 189
#         #
#         self.p3_1 = nn.Sequential(
#                                   # CausalDilationConv1d(16, 32, kernel_size=3, stride=1, dilation=5),
#                                   nn.Conv1d(16, 32, 3, 1, dilation=5),
#                                   nn.BatchNorm1d(32),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p3_2 = nn.Sequential(
#                                   # ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=2),
#                                   nn.Conv1d(32, 64, 3, 1, padding=2),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p3_3 = nn.Sequential(
#                                   # ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#                                   nn.Conv1d(64, 64, 3, 1, padding=1),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   ECA_block(64),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#
#         self.p4_1 = nn.Sequential(
#                                   # CausalDilationConv1d(16, 32, kernel_size=3, stride=1, dilation=3),
#                                   nn.Conv1d(16, 32, 3, 1, dilation=3),
#                                   nn.BatchNorm1d(32),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p4_2 = nn.Sequential(
#                                   # ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#                                   nn.Conv1d(32, 64, 3, 1, padding=1),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p4_3 = nn.Sequential(
#                                   # ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#                                   nn.Conv1d(64, 64, 3, 1, padding=1),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   ECA_block(64),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#
#         self.p5_1 = nn.Sequential(
#                                   # CausalDilationConv1d(16, 32, kernel_size=3, stride=1, dilation=1),
#                                   nn.Conv1d(16, 32, 3, 1, dilation=1),
#                                   nn.BatchNorm1d(32),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p5_2 = nn.Sequential(
#                                   # ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#                                   nn.Conv1d(32, 64, 3, 1, padding=1),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         self.p5_3 = nn.Sequential(
#                                   # ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#                                   nn.Conv1d(64, 64, 3, 1, padding=1),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   ECA_block(64),
#                                   nn.MaxPool1d(2)
#                                   )
#         # self
#         self.fc1 = nn.Linear(960, 100)
#         # self.fc1 = nn.Linear(512, 100)
#         self.eca = ECA_block(100)
#         self.relu1 = nn.LeakyReLU()
#         self.dp = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(100, 10)
#     #     self._initialize_weights()
#     #
#     # def _initialize_weights(self):
#     #     for m in self.modules():
#     #         if isinstance(m, nn.Conv1d):
#     #             nn.init.kaiming_normal_(m.weight.data)
#     #             nn.init.constant_(m.bias.data, 0.0)
#     #         elif isinstance(m, nn.BatchNorm1d):
#     #             m.weight.data.fill_(1)
#     #             m.bias.data.fill_(0)
#     #         elif isinstance(m, nn.Linear):
#     #             m.weight.data.normal_(0, 0.01)
#     #             if m.bias is not None:
#     #                 m.bias.data.fill_(1)
#
#     def forward(self, x):
#         x = self.p1(x)
#         p3 = self.p3_3(self.p3_2(self.p3_1(x)))
#         p4 = self.p4_3(self.p4_2(self.p4_1(x)))
#         p5 = self.p5_3(self.p5_2(self.p5_1(x)))
#         p1 = self.p1_1(x)
#         p1 = self.p1_2(p1)
#         # p1 = self.p1_3(p1)
#         # p1 = self.p1_4(p1)
#         # p1 = self.p1_5(p1)
#         p1out = self.p2_4(self.p2_3(self.p2_2(p1 + self.p2_1(p1))))
#         # p1out = self.p2_4(self.p2_3(p1 + self.p2_1(p1)))
#         # out1 = torch.cat((p1out, torch.cat((p5, torch.cat((p3, p4), dim=2)), dim=2)), dim=2)
#         # out1 = torch.cat((p1out, torch.cat((p4, p5), dim=2)), dim=2)
#         out = torch.add(p1out, torch.add(p5, torch.add(p4, p3)))
#         out = self.eca(out)
#         out = self.fc1(out.view(x.size(0), -1))
#         # out = self.fc1(out)
#         out = self.relu1(out)
#         out = self.dp(out)
#         out = self.fc2(out)
#         # return F.softmax(out, dim=1)
#         return out


# class mymodel1_4(nn.Module):
#     """
#     QCNN builder
#     """
#
#     def __init__(self, ) -> object:
#         super(mymodel1_4, self).__init__()
#         self.p1 = nn.Sequential(ConvQuadraticOperation(1, 16, kernel_size=64, stride=8, padding=28),
#                                 # nn.Conv1d(1, 16, 3, 1, 1),
#                                 nn.BatchNorm1d(16),
#                                 nn.Dropout(0.1),
#                                 nn.LeakyReLU(),
#                                 nn.MaxPool1d(2, 2)
#                                 )
#
#         self.p1_1 = nn.Sequential(ConvQuadraticOperation(16, 32, kernel_size=3, stride=1, padding=1),
#                                 # nn.Conv1d(16, 32, 3, 1, 1),
#                                 nn.BatchNorm1d(32),
#                                 nn.Dropout(0.1),
#                                 nn.LeakyReLU(),
#                                 nn.MaxPool1d(2, 2)
#                                 )
#         self.p1_2 = nn.Sequential(ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#                                 # nn.Conv1d(32, 64, 3, 1, 1),
#                                 nn.BatchNorm1d(64),
#                                 nn.Dropout(0.1),
#                                 nn.LeakyReLU(),
#                                 nn.MaxPool1d(2, 2)
#                                 )
#         # self.p1_3 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#         #                         # nn.Conv1d(64, 64, 3, 1, 1),
#         #                         nn.BatchNorm1d(64),
#         #                         nn.ReLU(),
#         #                         nn.MaxPool1d(2, 2)
#         #                         )
#         # self.p1_4 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#         #                           # nn.Conv1d(64, 64, 3, 1, 1),
#         #                           nn.BatchNorm1d(64),
#         #                           nn.ReLU(),
#         #                           nn.MaxPool1d(2, 2)
#         #                           )
#         # self.p1_5 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=0),
#         #                           # nn.Conv1d(64, 64, 3, 1, 1),
#         #                           nn.BatchNorm1d(64),
#         #                           nn.ReLU(),
#         #                           nn.MaxPool1d(2, 2)
#         #                           )
#         # 阈值降噪
#         self.p2_1 = sage(channel=64, gap_size=1)  # , 64, 1551
#         self.p2_2 = ECA_block(64)
#         # self.p2_3 = nn.Sequential(ConvQuadraticOperation(64, 64, 32, 4, 48),
#         self.p2_3 = nn.Sequential(ConvQuadraticOperation(64, 64, 32, 4, 48),
#         # self.p2_3 = nn.Sequential(ConvQuadraticOperation(64, 64, 32, 4, 32),
#                                   # nn.Conv1d(64, 32, 43, 4, bias=True),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU()
#                                   )  # , 32, 378
#         self.p2_4 = nn.MaxPool1d(2)         # , 32, 189
#
#         self.p3_1 = nn.Sequential(
#                                   # CausalDilationConv1d(16, 32, kernel_size=3, stride=1, dilation=5),
#                                   nn.Conv1d(16, 32, 3, 1, dilation=5, padding=5),
#                                   nn.BatchNorm1d(32),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         # self.p3_2 = nn.Sequential(
#         #                           # ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#         #                           nn.Conv1d(32, 64, 3, 1, padding=1),
#         #                           nn.BatchNorm1d(64),
#         #                           nn.Dropout(0.1),
#         #                           nn.LeakyReLU(),
#         #                           nn.MaxPool1d(2, 2)
#         #                           )
#         # self.p3_3 = nn.Sequential(
#         #                           # ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#         #                           nn.Conv1d(64, 64, 3, 1, padding=1),
#         #                           nn.BatchNorm1d(64),
#         #                           nn.Dropout(0.1),
#         #                           nn.LeakyReLU(),
#         #                           ECA_block(64),
#         #                           nn.MaxPool1d(2, 2)
#         #                           )
#
#         self.p4_1 = nn.Sequential(
#                                   # CausalDilationConv1d(16, 32, kernel_size=3, stride=1, dilation=3),
#                                   nn.Conv1d(16, 32, 3, 1, dilation=3, padding=3),
#                                   nn.BatchNorm1d(32),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         # self.p4_2 = nn.Sequential(
#         #                           # ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#         #                           nn.Conv1d(32, 64, 3, 1, padding=1),
#         #                           nn.BatchNorm1d(64),
#         #                           nn.Dropout(0.1),
#         #                           nn.LeakyReLU(),
#         #                           nn.MaxPool1d(2, 2)
#         #                           )
#         # self.p4_3 = nn.Sequential(
#         #                           # ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#         #                           nn.Conv1d(64, 64, 3, 1, padding=1),
#         #                           nn.BatchNorm1d(64),
#         #                           nn.Dropout(0.1),
#         #                           nn.LeakyReLU(),
#         #                           ECA_block(64),
#         #                           nn.MaxPool1d(2, 2)
#         #                           )
#
#         self.p5_1 = nn.Sequential(
#                                   # CausalDilationConv1d(16, 32, kernel_size=3, stride=1, dilation=1),
#                                   nn.Conv1d(16, 32, 3, 1, dilation=1, padding=1),
#                                   nn.BatchNorm1d(32),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#         # self.p5_2 = nn.Sequential(
#         #                           # ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#         #                           nn.Conv1d(32, 64, 3, 1, padding=1),
#         #                           nn.BatchNorm1d(64),
#         #                           nn.Dropout(0.1),
#         #                           nn.LeakyReLU(),
#         #                           nn.MaxPool1d(2, 2)
#         #                           )
#         # self.p5_3 = nn.Sequential(
#         #                           # ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
#         #                           nn.Conv1d(64, 64, 3, 1, padding=1),
#         #                           nn.BatchNorm1d(64),
#         #                           nn.Dropout(0.1),
#         #                           nn.LeakyReLU(),
#         #                           ECA_block(64),
#         #                           nn.MaxPool1d(2)
#         #                           )
#
#         self.p6_1 = nn.Sequential(ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
#                                   # nn.Conv1d(16, 32, 3, 1, 1),
#                                   nn.BatchNorm1d(64),
#                                   nn.Dropout(0.1),
#                                   nn.LeakyReLU(),
#                                   nn.MaxPool1d(2, 2)
#                                   )
#
#         self.fc1 = nn.Linear(2048, 100)
#         # self.fc1 = nn.Linear(512, 100)
#         self.eca = ECA_block(100)
#         self.relu1 = nn.LeakyReLU()
#         self.dp = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(100, 10)
#
#
#     def forward(self, x):
#         x = self.p1(x)
#         # p3 = self.p3_3(self.p3_2(self.p3_1(x)))
#         # p4 = self.p4_3(self.p4_2(self.p4_1(x)))
#         # p5 = self.p5_3(self.p5_2(self.p5_1(x)))
#         p3 = self.p3_1(x)
#         p4 = self.p4_1(x)
#         p5 = self.p5_1(x)
#         p1 = self.p1_1(x)
#         p1 = self.p1_2(p1)
#         # p1out = self.p2_4(self.p2_3(self.p2_2(p1 + self.p2_1(p1))))
#         out = torch.add(p5, torch.add(p4, p3))
#         # out = torch.concat((p5, torch.concat((p4, p3),dim=2)), dim=2)
#         out = self.p6_1(out)
#         out = torch.add(p1, out)
#         out = self.eca(out)
#         out = self.fc1(out.view(x.size(0), -1))
#         # out = self.fc1(out)
#         out = self.relu1(out)
#         out = self.dp(out)
#         out = self.fc2(out)
#         # return F.softmax(out, dim=1)
#         return out


class mymodel1_4(nn.Module):
    """
    QCNN builder
    """

    def __init__(self, ) -> object:
        super(mymodel1_4, self).__init__()
        self.p1 = nn.Sequential(ConvQuadraticOperation(1, 16, kernel_size=64, stride=8, padding=28),
                                # nn.Conv1d(1, 16, 3, 1, 1),
                                nn.BatchNorm1d(16),
                                nn.Dropout(0.1),
                                nn.LeakyReLU(),
                                nn.MaxPool1d(2, 2)
                                )

        self.p1_1 = nn.Sequential(ConvQuadraticOperation(16, 32, kernel_size=3, stride=1, padding=1),
                                # nn.Conv1d(16, 32, 3, 1, 1),
                                nn.BatchNorm1d(32),
                                nn.Dropout(0.1),
                                nn.LeakyReLU(),
                                # nn.MaxPool1d(2, 2)
                                )
        self.p1_2 = nn.Sequential(ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
                                # nn.Conv1d(32, 64, 3, 1, 1),
                                nn.BatchNorm1d(64),
                                nn.Dropout(0.1),
                                nn.LeakyReLU(),
                                # nn.MaxPool1d(2, 2)
                                )
        # self.p1_3 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
        #                         # nn.Conv1d(64, 64, 3, 1, 1),
        #                         nn.BatchNorm1d(64),
        #                         nn.ReLU(),
        #                         nn.MaxPool1d(2, 2)
        #                         )
        # self.p1_4 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=1),
        #                           # nn.Conv1d(64, 64, 3, 1, 1),
        #                           nn.BatchNorm1d(64),
        #                           nn.ReLU(),
        #                           nn.MaxPool1d(2, 2)
        #                           )
        # self.p1_5 = nn.Sequential(ConvQuadraticOperation(64, 64, kernel_size=3, stride=1, padding=0),
        #                           # nn.Conv1d(64, 64, 3, 1, 1),
        #                           nn.BatchNorm1d(64),
        #                           nn.ReLU(),
        #                           nn.MaxPool1d(2, 2)
        #                           )
        # 阈值降噪
        self.p2_1 = sage(channel=64, gap_size=1)  # , 64, 1551
        self.p2_2 = ECA_block(64)
        # self.p2_3 = nn.Sequential(ConvQuadraticOperation(64, 64, 43, 4, 18),
        #                           # nn.Conv1d(64, 32, 43, 4, bias=True),
        #                           nn.BatchNorm1d(64),
        #                           nn.Dropout(0.1),
        #                           nn.LeakyReLU()
        #                           )         # , 32, 378
        self.p2_3 = nn.Sequential(ConvQuadraticOperation(64, 64, 32, 4, 16),
        # self.p2_3 = nn.Sequential(ConvQuadraticOperation(64, 64, 32, 4, 32),
                                  # nn.Conv1d(64, 32, 43, 4, bias=True),
                                  nn.BatchNorm1d(64),
                                  nn.Dropout(0.1),
                                  nn.LeakyReLU()
                                  )  # , 32, 378
        self.p2_4 = nn.MaxPool1d(2)         # , 32, 189

        self.p3_1 = nn.Sequential(
                                  CausalDilationConv1d(16, 32, kernel_size=3, stride=2, dilation=5),
                                  # nn.Conv1d(16, 32, 3, 1, dilation=5),
                                  nn.BatchNorm1d(32),
                                  nn.Dropout(0.1),
                                  nn.LeakyReLU(),
                                  nn.MaxPool1d(2, 2)
                                  )
        # self.p3_2 = nn.Sequential(
        #                           ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
        #                           # nn.Conv1d(32, 64, 3, 1, padding=1),
        #                           nn.BatchNorm1d(64),
        #                           nn.Dropout(0.1),
        #                           nn.LeakyReLU(),
        #                           nn.MaxPool1d(2, 2)
        #                           )
        self.p3_3 = nn.Sequential(
                                  ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
                                  # nn.Conv1d(64, 64, 3, 1, padding=1),
                                  nn.BatchNorm1d(64),
                                  nn.Dropout(0.1),
                                  nn.LeakyReLU(),
                                  ECA_block(64),
                                  nn.MaxPool1d(2, 2)
                                  )

        self.p4_1 = nn.Sequential(
                                  CausalDilationConv1d(16, 32, kernel_size=3, stride=2, dilation=3),
                                  # nn.Conv1d(16, 32, 3, 1, dilation=3),
                                  nn.BatchNorm1d(32),
                                  nn.Dropout(0.1),
                                  nn.LeakyReLU(),
                                  nn.MaxPool1d(2, 2)
                                  )
        # self.p4_2 = nn.Sequential(
        #                           ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
        #                           # nn.Conv1d(32, 64, 3, 1, padding=1),
        #                           nn.BatchNorm1d(64),
        #                           nn.Dropout(0.1),
        #                           nn.LeakyReLU(),
        #                           nn.MaxPool1d(2, 2)
        #                           )
        self.p4_3 = nn.Sequential(
                                  ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
                                  # nn.Conv1d(64, 64, 3, 1, padding=1),
                                  nn.BatchNorm1d(64),
                                  nn.Dropout(0.1),
                                  nn.LeakyReLU(),
                                  ECA_block(64),
                                  nn.MaxPool1d(2, 2)
                                  )

        self.p5_1 = nn.Sequential(
                                  CausalDilationConv1d(16, 32, kernel_size=3, stride=2, dilation=1),
                                  # nn.Conv1d(16, 32, 3, 1, dilation=1),
                                  nn.BatchNorm1d(32),
                                  nn.Dropout(0.1),
                                  nn.LeakyReLU(),
                                  nn.MaxPool1d(2, 2)
                                  )
        # self.p5_2 = nn.Sequential(
        #                           ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
        #                           # nn.Conv1d(32, 64, 3, 1, padding=1),
        #                           nn.BatchNorm1d(64),
        #                           nn.Dropout(0.1),
        #                           nn.LeakyReLU(),
        #                           nn.MaxPool1d(2, 2)
        #                           )
        self.p5_3 = nn.Sequential(
                                  ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
                                  # nn.Conv1d(64, 64, 3, 1, padding=1),
                                  nn.BatchNorm1d(64),
                                  nn.Dropout(0.1),
                                  nn.LeakyReLU(),
                                  ECA_block(64),
                                  nn.MaxPool1d(2, 2)
                                  )
        # self
        self.eca = ECA_block(100)
        self.fc1 = nn.Linear(1024, 100)
        # self.fc1 = nn.Linear(512, 100)
        self.relu1 = nn.LeakyReLU()
        self.dp = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, 10)
    #     self._initialize_weights()
    #
    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv1d):
    #             nn.init.kaiming_normal_(m.weight.data)
    #             nn.init.constant_(m.bias.data, 0.0)
    #         elif isinstance(m, nn.BatchNorm1d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.fill_(0)
    #         elif isinstance(m, nn.Linear):
    #             m.weight.data.normal_(0, 0.01)
    #             if m.bias is not None:
    #                 m.bias.data.fill_(1)

    def forward(self, x):
        x = self.p1(x)
        p3 = self.p3_3(self.p3_1(x))
        p4 = self.p4_3(self.p4_1(x))
        p5 = self.p5_3(self.p5_1(x))
        p1 = self.p1_1(x)
        p1 = self.p1_2(p1)
        p1out = self.p2_4(self.p2_3(self.p2_2(p1 + self.p2_1(p1))))
        out = torch.add(p1out, torch.add(p5, torch.add(p4, p3)))
        out = self.eca(out)
        out = self.fc1(out.view(x.size(0), -1))
        out = self.relu1(out)
        out = self.dp(out)
        out = self.fc2(out)
        return out

    # def forward(self, x):
    #     x = self.p1(x)
    #     # p3 = self.p3_3(self.p3_2(self.p3_1(x)))
    #     # p4 = self.p4_3(self.p4_2(self.p4_1(x)))
    #     # p5 = self.p5_3(self.p5_2(self.p5_1(x)))
    #     p3 = self.p3_3(self.p3_1(x))
    #     p4 = self.p4_3(self.p4_1(x))
    #     p5 = self.p5_3(self.p5_1(x))
    #     # p3 = self.p3_1(x)
    #     # p4 = self.p4_1(x)
    #     # p5 = self.p5_1(x)
    #     p1 = self.p1_1(x)
    #     p1 = self.p1_2(p1)
    #     p1out = self.p2_4(self.p2_3(self.p2_2(p1 + self.p2_1(p1))))
    #     # p1out = self.p2_4(self.p2_3(p1 + self.p2_1(p1)))
    #     # pout = torch.add(p5, torch.add(p4, p3))
    #     # pout = self.p3_3(self.p3_2(pout))
    #     out = torch.add(p1out, torch.add(p5, torch.add(p4, p3)))
    #     # out = torch.add(p5, torch.add(p4, p3))
    #     # out = torch.add(pout, p1out)
    #     out = self.eca(out)
    #     out = self.fc1(out.view(x.size(0), -1))
    #     out = self.relu1(out)
    #     out = self.dp(out)
    #     out = self.fc2(out)
    #     return out

if __name__ == '__main__':
    X = torch.rand(64, 1, 2048).cuda()
    m = mymodel1_4()
    summary(m.cuda(), (1, 2048))
    print(flop_count_str(FlopCountAnalysis(m.cuda(), X)))