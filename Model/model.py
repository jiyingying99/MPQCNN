import torch
from torch import nn
from torchsummary import summary
from fvcore.nn import FlopCountAnalysis, flop_count_str
from Model.ConvQuadraticOperation import ConvQuadraticOperation
from Model.ECA import ECA_block
from Model.Shrinkage import Shrinkage as sage
from Model.CausalDilationConv1d import CausalDilationConv1d

class mymodel1_4(nn.Module):
    """
    QCNN builder
    """

    def __init__(self, ) -> object:
        super(mymodel1_4, self).__init__()
        self.p1 = nn.Sequential(ConvQuadraticOperation(1, 16, kernel_size=64, stride=8, padding=28),
                                nn.BatchNorm1d(16),
                                nn.Dropout(0.1),
                                nn.LeakyReLU(),
                                nn.MaxPool1d(2, 2)
                                )
        self.p1_1 = nn.Sequential(ConvQuadraticOperation(16, 32, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm1d(32),
                                nn.Dropout(0.1),
                                nn.LeakyReLU()
                                )
        self.p1_2 = nn.Sequential(ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm1d(64),
                                nn.Dropout(0.1),
                                nn.LeakyReLU(),
                                )
        # 阈值降噪
        self.p2_1 = sage(channel=64, gap_size=1)  # , 64, 1551
        self.p2_2 = ECA_block(64)
        self.p2_3 = nn.Sequential(ConvQuadraticOperation(64, 64, 32, 4, 16),
                                  nn.BatchNorm1d(64),
                                  nn.Dropout(0.1),
                                  nn.LeakyReLU()
                                  )
        self.p2_4 = nn.MaxPool1d(2)

        self.p3_1 = nn.Sequential(
                                  CausalDilationConv1d(16, 32, kernel_size=3, stride=2, dilation=5),
                                  nn.BatchNorm1d(32),
                                  nn.Dropout(0.1),
                                  nn.LeakyReLU(),
                                  nn.MaxPool1d(2, 2)
                                  )
        self.p3_3 = nn.Sequential(
                                  ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
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
        self.p5_3 = nn.Sequential(
                                  ConvQuadraticOperation(32, 64, kernel_size=3, stride=1, padding=1),
                                  # nn.Conv1d(64, 64, 3, 1, padding=1),
                                  nn.BatchNorm1d(64),
                                  nn.Dropout(0.1),
                                  nn.LeakyReLU(),
                                  ECA_block(64),
                                  nn.MaxPool1d(2, 2)
                                  )
        self.eca = ECA_block(100)
        self.fc1 = nn.Linear(1024, 100)
        self.relu1 = nn.LeakyReLU()
        self.dp = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, 10)

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

if __name__ == '__main__':
    X = torch.rand(64, 1, 2048).cuda()
    m = mymodel1_4()
    summary(m.cuda(), (1, 2048))
    print(flop_count_str(FlopCountAnalysis(m.cuda(), X)))