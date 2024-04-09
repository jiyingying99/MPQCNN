import torch
from torch import nn

class Shrinkage(nn.Module):   #半软阈值降噪
    def __init__(self, channel, gap_size):
        super(Shrinkage, self).__init__()
        self.a = nn.Parameter(torch.tensor([0.48]))     # 0.48
        self.gap = nn.AdaptiveAvgPool1d(gap_size)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid()
        )

    def forward(self, x):

        x_raw = x       # x_raw: tensor:(1, 64, 256)
        x_abs = x.abs()     # x_abs: tensor:(1, 64, 256)
        x = self.gap(x)     # x: tensor:(1, 64, 2)
        x = torch.flatten(x, 1)     # x: tensor:(1, 128)        128=64*2
        average = x     # average: tensor(1, 128)
        # average = torch.mean(x, dim=1, keepdim=True)
        # x = self.fc(out.view(x.size(0), -1))
        # if x.size(0) > 1:
        #     x = self.fc(x)
        # else:
        #     # x = self.fc(x.view(x.size(1), -1))
        #     x = self.fc(x.view(x.size(0), -1)).view(x.size(0), -1, 1)
        x = self.fc(x)
        x = torch.mul(average, x).unsqueeze(2)
        # soft thresholding
        sub = torch.max(x_abs - x, torch.zeros_like(x_abs - x))
        mask = sub.clone()
        mask[mask > 0] = 1
        # a = torch.clamp(self.a, min=0, max=1)
        x = sub + (1 - self.a) * x
        x = torch.mul(torch.sign(x_raw), torch.mul(x, mask))

        return x



if __name__ == '__main__':
    input = torch.randn(2, 64, 256)
    model = Shrinkage(channel=64, gap_size=1)
    # model = Shrinkage(1, 64)
    for param in model.parameters():
        print(type(param.data), param.size())
    output = model(input)
    print(output.shape)