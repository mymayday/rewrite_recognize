from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.cuda import is_available

import math
from torch.nn import Parameter
from config import configer

class Bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expansion):
        super(Bottleneck, self).__init__()
        self.connect = stride == 1 and inp == oup
        
        self.conv = nn.Sequential(
            #pointwise
            nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # nn.ReLU(inplace=True),

            #depthwise conv3x3
            nn.Conv2d(inp * expansion, inp * expansion, 3, stride, 1, groups=inp * expansion, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # nn.ReLU(inplace=True),

            #pointwise-linear
            nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        #Bottleneck的Block中步长为1时采用shortcut,步长为2时则不采用
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)
    def forward(self, x):    
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)

Mobilefacenet_bottleneck_setting = [
    # t, c , n ,s
    [2, 64, 5, 2],
    [4, 128, 1, 2],
    [2, 128, 6, 1],
    [4, 128, 1, 2],
    [2, 128, 2, 1]
]

Mobilenetv2_bottleneck_setting = [
    # t, c, n, s
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
]

class MobileFacenet(nn.Module):
    def __init__(self, bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(MobileFacenet, self).__init__()
        
        #第一层 standard convolution conv 3x3
        self.conv1 = ConvBlock(configer.n_usedChannels, 64, 3, 2, 1)
        #第二层 depthwise convolution 3x3
        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)
        self.inplanes = 64
        block = Bottleneck
        self.blocks = self._make_layer(block, bottleneck_setting)
        #conv1x1
        self.conv2 = ConvBlock(128, 512, 1, 1, 0)
        #linear GDConv
        k_size = (MobileFacenet.get_input_res()[0] // 16, MobileFacenet.get_input_res()[1] // 16)
        self.linear7 = ConvBlock(512, 512, k_size, 1, 0, dw=True, linear=True)
        #linear Conv1x1
        self.linear1 = ConvBlock(512, 128, 1, 1, 0, linear=True)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        


    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            #Bottleneck的第一个Block的步长为表中设置的步长,从第二个Block开始的步长都为1
            for i in range(n):
                if i == 0:
                    layers.append(block(self.inplanes, c, s, t))            
                else:
                    layers.append(block(self.inplanes, c, 1, t))
                self.inplanes = c

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.linear7(x)
        x = self.linear1(x)
        x = x.contiguous().view(x.size(0), -1)
                
        return x
    
    @staticmethod
    def get_input_res():
        return 64,64

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features=128, out_features=63, s=32.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        # if configer.cuda and is_available():
        #     self.weight = Parameter(torch.cuda.FloatTensor(out_features, in_features))
        # nn.init.xavier_uniform_(self.weight)
if configer.cuda and is_available():
        #     self.weight = Parameter(torch.cuda.FloatTensor(out_features, in_features))
        # nn.init.xavier_uniform_(self.weight)        # init.kaiming_uniform_()
        # self.weight.data.normal_(std=0.001)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.contiguous().view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output




# if __name__ == "__main__":
#     input = Variable(torch.FloatTensor(2, 3, 64,64))
#     net = MobileFacenet()
#     print(net)
#     x = net(input)
#     print(x.shape)
