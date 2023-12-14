import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import numpy as np


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim=81, num_heads=4):
        super().__init__()
        self.dim = dim
        self.scale = self.dim ** -0.5
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

        # Orthogonal initialization of weights
        init.orthogonal_(self.qkv.weight)
        init.orthogonal_(self.proj.weight)

        # Constant initialization of biases
        if self.qkv.bias is not None:
            init.constant_(self.qkv.bias, 0)
        if self.proj.bias is not None:
            init.constant_(self.proj.bias, 0)

    def forward(self, inputs, mask=None):
        qkv = self.qkv(inputs).reshape(
            inputs.shape[0], -1, 4, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask = mask[:, None, None, :]

        attn = F.softmax(attn, dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(inputs.shape[0], -1, self.dim)
        x = self.proj(x)
        return x


class ConvCNN(nn.Module):
    def __init__(self, insize, outsize, kernel_size=7, padding=2, pool=2, avg=True):
        super(ConvCNN, self).__init__()
        self.avg = avg
        self.math = torch.nn.Sequential(
            torch.nn.Conv2d(insize, outsize,
                            kernel_size=kernel_size, padding=padding),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(pool, pool),
        )
        self.avgpool = torch.nn.AvgPool2d(pool, pool)

    def forward(self, x):
        x = self.math(x)
        if self.avg is True:
            x = self.avgpool(x)
        return x


class Block(nn.Module):
    '''Grouped convolution block.'''
    expansion = 2

    def __init__(self, in_planes, cardinality=32, bottleneck_width=4, stride=1):
        super(Block, self).__init__()
        group_width = cardinality * bottleneck_width
        self.conv1 = nn.Conv2d(in_planes, group_width,
                               kernel_size=1, bias=False)

        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality,
                               bias=False)

        self.conv3 = nn.Conv2d(
            group_width, self.expansion * group_width, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * group_width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * group_width,
                          kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu((self.conv1(x)))
        out = F.relu((self.conv2(out)))
        out = (self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNeXt(nn.Module):
    def __init__(self, num_blocks, cardinality, bottleneck_width, num_classes=256):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64

        self.conv1 = nn.Conv2d(96, 64, kernel_size=1, bias=False)

        self.layer1 = self._make_layer(num_blocks[0], 1)
        self.layer2 = self._make_layer(num_blocks[1], 2)
        self.layer3 = self._make_layer(num_blocks[2], 2)
        # self.layer4 = self._make_layer(num_blocks[3], 2)

        # self.linear = nn.Linear(cardinality*bottleneck_width*8, num_classes)
        # self.linear = nn.Linear(3840, num_classes)
        self.activation = torch.nn.ReLU()
        self.sig = nn.Sigmoid()

    def _make_layer(self, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(Block(self.in_planes, self.cardinality,
                          self.bottleneck_width, stride))
            self.in_planes = Block.expansion * self.cardinality * self.bottleneck_width
        # Increase bottleneck_width by 2 after each stage.
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu((self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(x.size(0), -1)
        # print (out.data.shape)
        out = self.activation(out)
        # out = F.log_softmax(out)
        # out = self.sig(out)
        return out


def ResNeXt29_2x64d():
    """
    https://www.kaggle.com/code/solomonk/pytorch-resnext-cnn-end-to-end-lb-0-65?scriptVersionId=1872910&cellId=2
    """
    return ResNeXt(num_blocks=[1, 1, 1], cardinality=4, bottleneck_width=8)


class TransformerBlock(nn.Module):
    def __init__(self, dim=81, num_heads=3, expand=1, activation=F.relu):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MultiHeadSelfAttention(dim=dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.fc1 = nn.Linear(dim, dim * expand, )
        self.fc2 = nn.Linear(dim * expand, dim, )
        self.activation = activation
        # Orthogonal initialization of weights
        init.orthogonal_(self.fc1.weight)
        init.orthogonal_(self.fc2.weight)

        # # Constant initialization of biases
        # if self.fc1.bias is not None:
        #     init.constant_(self.fc1.bias, 0)
        # if self.fc2.bias is not None:
        #     init.constant_(self.fc2.bias, 0)

    def forward(self, inputs):
        x = self.norm1(inputs)
        x = self.attn(x)
        attn_out = x

        x = self.norm2(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = x + attn_out
        return x


class PopArt(torch.nn.Module):

    def __init__(self, input_shape, output_shape, norm_axes=1, beta=0.99999, epsilon=1e-5, device=torch.device("cpu")):

        super(PopArt, self).__init__()

        self.beta = beta
        self.epsilon = epsilon
        self.norm_axes = norm_axes
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.weight = nn.Parameter(torch.Tensor(
            output_shape, input_shape)).to(**self.tpdv)
        self.bias = nn.Parameter(torch.Tensor(output_shape)).to(**self.tpdv)

        self.stddev = nn.Parameter(torch.ones(
            output_shape), requires_grad=False).to(**self.tpdv)
        self.mean = nn.Parameter(torch.zeros(
            output_shape), requires_grad=False).to(**self.tpdv)
        self.mean_sq = nn.Parameter(torch.zeros(
            output_shape), requires_grad=False).to(**self.tpdv)
        self.debiasing_term = nn.Parameter(torch.tensor(
            0.0), requires_grad=False).to(**self.tpdv)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
                self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        self.mean.zero_()
        self.mean_sq.zero_()
        self.debiasing_term.zero_()

    def forward(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        return F.linear(input_vector, self.weight, self.bias)

    @torch.no_grad()
    def update(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        old_mean, old_var = self.debiased_mean_var()
        old_stddev = torch.sqrt(old_var)

        batch_mean = input_vector.mean(dim=tuple(range(self.norm_axes)))
        batch_sq_mean = (
            input_vector ** 2).mean(dim=tuple(range(self.norm_axes)))

        self.mean.mul_(self.beta).add_(batch_mean * (1.0 - self.beta))
        self.mean_sq.mul_(self.beta).add_(batch_sq_mean * (1.0 - self.beta))
        self.debiasing_term.mul_(self.beta).add_(1.0 * (1.0 - self.beta))

        self.stddev = (self.mean_sq - self.mean ** 2).sqrt().clamp(min=1e-4)

        new_mean, new_var = self.debiased_mean_var()
        new_stddev = torch.sqrt(new_var)

        self.weight = self.weight * old_stddev / new_stddev
        self.bias = (old_stddev * self.bias + old_mean - new_mean) / new_stddev

    def debiased_mean_var(self):
        debiased_mean = self.mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.mean_sq / \
            self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    def normalize(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.debiased_mean_var()
        out = (input_vector - mean[(None,) * self.norm_axes]
               ) / torch.sqrt(var)[(None,) * self.norm_axes]

        return out

    def denormalize(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.debiased_mean_var()
        out = input_vector * \
            torch.sqrt(var)[(None,) * self.norm_axes] + \
            mean[(None,) * self.norm_axes]

        out = out.cpu().numpy()

        return out
