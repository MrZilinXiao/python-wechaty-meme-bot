"""
Reimplementation of `Deep Cosine Metric Learning for Person Re-identification`
Model of original paper was written in Tensorflow-Slim;
Here I implement it in PyTorch, but no further test yet.
Link: https://ieeexplore.ieee.org/document/8354191/
Citation:
[1]N. Wojke and A. Bewley, “Deep Cosine Metric Learning for Person Re-identification,”
in 2018 IEEE Winter Conference on Applications of Computer Vision (WACV),
Lake Tahoe, NV, Mar. 2018, pp. 748–756, doi: 10.1109/WACV.2018.00087.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import pad
from torch.nn.modules.utils import _pair
import math


class _ConvNd(nn.Module):
    """
    Copied from torch source code
    since Conv2d_with_padding needs to derived from it and it's a private class.
    """
    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode='zeros'):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        if transposed:
            self.weight = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_ConvNd, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'


class Conv2d_with_padding(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d_with_padding, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    @staticmethod
    def conv2d_same_padding(input, weight, bias=None, stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1):
        # padding not relevant, just to be compatible with conv2d
        input_rows = input.size(2)
        filter_rows = weight.size(2)
        effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
        out_rows = (input_rows + stride[0] - 1) // stride[0]
        padding_rows = max(0, (out_rows - 1) * stride[0] +
                           (filter_rows - 1) * dilation[0] + 1 - input_rows)
        rows_odd = (padding_rows % 2 != 0)
        padding_cols = max(0, (out_rows - 1) * stride[0] +
                           (filter_rows - 1) * dilation[0] + 1 - input_rows)
        cols_odd = (padding_rows % 2 != 0)

        if rows_odd or cols_odd:
            input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])

        return F.conv2d(input, weight, bias, stride,
                        padding=(padding_rows // 2, padding_cols // 2),
                        dilation=dilation, groups=groups)

    def forward(self, input):
        return Conv2d_with_padding.conv2d_same_padding(input, self.weight, self.bias, self.stride,
                                                       self.padding, self.dilation, self.groups)


class CosineMetricNet(nn.Module):
    feature_dim = 128

    def __init__(self, num_classes, add_logits=True):
        """
        Cosine Metric Net defined in `Deep Cosine Metric Learning for Person Re-identification`
        However, using Triplet Loss seems to be not effective
        Input: N X 3 X 128 X 64
        Output: N X 128 with L2 regularization
        """
        super(CosineMetricNet, self).__init__()
        self.add_logits = add_logits  # whether using CrossEntropy Loss
        self.num_classes = num_classes
        self.res_list = [
            [1, [32, 32], True],  # format: [stride, [in_channels, out_channels], flag]
            [1, [32, 32], True],
            [2, [32, 64], False],
            [1, [64, 64], True],
            [2, [64, 128], False],
            [1, [128, 128], True]
        ]

        # residual 4~9
        # self.res_part = nn.ModuleList([
        #     ResidualBlock(32, 32, 1),
        #     ResidualBlock(32, 32, 1),
        #     ResidualBlock(32, 64, 2, False),
        #     ResidualBlock(64, 64, 1),
        #     ResidualBlock(64, 128, 2, False),
        #     ResidualBlock(128, 128, 1)
        # ])

        self.res_part = nn.ModuleList([
            ResidualBlock(in_channels, out_channels, stride, flag)
            for stride, (in_channels, out_channels), flag in self.res_list
        ])

        self.net = nn.Sequential(
            Conv2d_with_padding(in_channels=3, out_channels=32, kernel_size=3, stride=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(32),
            # conv_1

            Conv2d_with_padding(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(32),
            # conv_2

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(32),
            # maxpool_3
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.6),
            nn.Linear(in_features=128 * 16 * 8, out_features=128),
        )
        if self.add_logits:
            self.weights = torch.FloatTensor(CosineMetricNet.feature_dim, int(self.num_classes))
            self.weights = self._truncated_normal_(self.weights, std=1e-3)
            self.weights = nn.Parameter(self.weights, requires_grad=True)
            self.scale = torch.FloatTensor([0])
            self.scale = nn.Parameter(self.scale, requires_grad=True)

    def _truncated_normal_(self, tensor, mean=0, std=0.09):
        """
        Init a truncated normal tensor like tf.truncated_normal_initializer does.
        Come from: https://zhuanlan.zhihu.com/p/83609874
        """
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def forward(self, input, add_logits=True):
        out = self.net(input)
        for resblock in self.res_part:
            out = resblock(out)
        out = self.linear(out)
        out = F.normalize(out, dim=1, p=2)
        if add_logits:
            scale = F.softplus(self.scale)
            weight_normed = F.normalize(self.weights, dim=0, p=2)
            logits = scale * torch.matmul(out, weight_normed)
            return out, logits
        else:
            return out

    def forward_twice(self, input1, input2):
        out1 = self.forward(input1, add_logits=False)
        out2 = self.forward(input2, add_logits=False)
        return out1, out2

    @staticmethod
    def _cosine_distance(a: torch.Tensor, b: torch.Tensor):
        assert a.shape == b.shape, "Tensor Shape Do Not Match"
        return torch.ones([a.shape[i] for i in range(len(a.shape)) if i != 1]) - F.cosine_similarity(a, b)  # ignore dim 1

    @staticmethod
    def validator():
        pass
    # def forward(self, input1, input2, input3=None):
    #     output1 = self.forward_once(input1)
    #     output2 = self.forward_once(input2)
    #     if input3 is not None:
    #         output3 = self.forward_once(input3)
    #         return output1, output2, output3
    #     return output1, output2


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1, same_shape=True):
        super(ResidualBlock, self).__init__()
        self.same_shape = same_shape
        if not same_shape:
            strides = 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ELU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        if not same_shape:
            self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = self.block(x)
        if not self.same_shape:
            x = self.bn3(self.conv3(x))
        # return F.relu(out + x)
        return out + x
