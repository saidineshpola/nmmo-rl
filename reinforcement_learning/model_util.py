import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


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
