import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance', d2=True):
    """Return a normalization layer
    """
    if norm_type is None or norm_type.lower() == 'none':
        norm_layer = lambda x: Identity()
    elif norm_type.lower() == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True) if d2 \
            else functools.partial(nn.BatchNorm1d, affine=True, track_running_stats=True)
    elif norm_type.lower() == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False) if d2 \
            else functools.partial(nn.InstanceNorm1d, affine=False, track_running_stats=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_nonlinear_layer(nonlinear_type='relu'):
    """Return a nonlinear layer
    """
    if nonlinear_type is None or nonlinear_type.lower() == 'none':
        return Identity
    elif nonlinear_type.lower() == 'leakyrelu':
        return functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=False)
    elif nonlinear_type.lower() == 'relu':
        return functools.partial(nn.ReLU, inplace=False)
    elif nonlinear_type.lower() == 'sigmoid':
        return nn.Sigmoid
    elif nonlinear_type.lower() == 'tanh':
        return nn.Tanh
    else:
        raise ValueError


class ECABlock(nn.Module):
    """Constructs a ECA module.
    Args:
        input_nc: Number of channels of the input feature map
        kernel_size: Adaptive selection of kernel size
    """

    def __init__(self, input_nc, reduction=16):
        super(ECABlock, self).__init__()
        gamma = 2.
        b = 1.
        t = int(abs(np.log2(input_nc) + b) / gamma)
        kernel_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class IdentityBlock(nn.Module):
    def __init__(self, input_nc, reduction):
        super().__init__()

    def forward(self, x):
        return x


def get_attention_layer(attention_type='eca'):
    if attention_type is None or attention_type.lower() == 'none':
        return IdentityBlock
    elif attention_type.lower() == 'eca':
        return ECABlock
    else:
        raise ValueError


class ConvGRUBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size=3, dilation=1, padding=1, conv_type='same', norm_type=None,
                 attention_type=None, reduction=16, nonlinear_type='leakyrelu'):
        super().__init__()
        self.conv_xz = Conv2DBlock(input_nc, output_nc, kernel_size, dilation=dilation, conv_type=conv_type,
                                   padding=padding, norm_type=norm_type)
        self.conv_xr = Conv2DBlock(input_nc, output_nc, kernel_size, dilation=dilation, conv_type=conv_type,
                                   padding=padding, norm_type=norm_type)
        self.conv_xn = Conv2DBlock(input_nc, output_nc, kernel_size, dilation=dilation, conv_type=conv_type,
                                   padding=padding, norm_type=norm_type)

        self.conv_hz = Conv2DBlock(output_nc, output_nc, kernel_size, conv_type='same',
                                   padding=padding, norm_type=norm_type)
        self.conv_hr = Conv2DBlock(output_nc, output_nc, kernel_size, conv_type='same',
                                   padding=padding, norm_type=norm_type)
        self.conv_hn = Conv2DBlock(output_nc, output_nc, kernel_size, conv_type='same',
                                   padding=padding, norm_type=norm_type)

        self.attention = get_attention_layer(attention_type)(output_nc, reduction)
        self.nonliearity = get_nonlinear_layer(nonlinear_type)()

    def forward(self, x, h=None, x_skip=None):
        xz_hat = self.conv_xz(x) if x_skip is None else self.conv_xz(x, x_skip.shape[-2:])
        xn_hat = self.conv_xn(x) if x_skip is None else self.conv_xn(x, x_skip.shape[-2:])
        if h is None:
            z = torch.sigmoid(xz_hat)
            f = torch.tanh(xn_hat)
            h = z * f
        else:
            xr_hat = self.conv_xr(x) if x_skip is None else self.conv_xr(x, x_skip.shape[-2:])
            z = torch.sigmoid(xz_hat + self.conv_hz(h))
            r = torch.sigmoid(xr_hat + self.conv_hr(h))
            n = torch.tanh(xn_hat + self.conv_hn(r * h))
            h = (1 - z) * h + z * n
        h = self.nonliearity(self.attention(h))
        return h if x_skip is None else upsample_cat(h, x_skip), h


def get_conv_rnn_layer(rnn_type='GRU'):
    conv_rnn_layer = {
        'gru': ConvGRUBlock
    }[rnn_type.lower()]
    return conv_rnn_layer


def get_conv2d_layer(conv_type='same'):
    if conv_type.lower() == 'same':
        conv_layer = functools.partial(nn.Conv2d, stride=1)
    elif conv_type.lower() == 'up':
        conv_layer = functools.partial(nn.ConvTranspose2d, stride=2)
    elif conv_type.lower() == 'down':
        conv_layer = functools.partial(nn.Conv2d, stride=2)
    else:
        raise ValueError
    return conv_layer


class Conv2DBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size, groups=1, bias=True, padding=None, dilation=1,
                 conv_type='same', norm_type=None, attention_type=None, reduction=16, nonlinear_type=None):
        super(Conv2DBlock, self).__init__()
        self.norm = get_norm_layer(norm_type)(output_nc) if norm_type is not None else None
        if self.norm is not None:
            if type(self.norm) == functools.partial:
                bias = self.norm.func == nn.InstanceNorm2d
            else:
                bias = self.norm == nn.InstanceNorm2d
        if padding is None:
            padding = (dilation * (kernel_size - 1) // 2)
        self.conv_type = conv_type
        self.conv = get_conv2d_layer(self.conv_type)(
            in_channels=input_nc, out_channels=output_nc, kernel_size=kernel_size, padding=padding,
            groups=groups, bias=bias, dilation=dilation) if self.conv_type is not None else None
        self.attention = get_attention_layer(attention_type)(output_nc,
                                                             reduction) if attention_type is not None else None
        self.nonliearity = get_nonlinear_layer(nonlinear_type)() if nonlinear_type is not None else None

    def forward(self, x, output_size=None):
        if self.conv:
            if output_size is not None:
                assert isinstance(self.conv, nn.ConvTranspose2d)
                x = self.conv(x, output_size=output_size)
            else:
                if self.conv_type == 'same':
                    x = conv2d_same_padding(x, self.conv)
                else:
                    x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.attention:
            x = self.attention(x)
        if self.nonliearity:
            x = self.nonliearity(x)
        return x


def conv2d_same_padding(x, conv):
    stride = conv.stride
    dilation = conv.dilation
    kernel_size = conv.kernel_size
    padding = conv.padding

    # calculating rows_odd
    input_rows = x.size(2)
    effective_filter_size_rows = (kernel_size[0] - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] + effective_filter_size_rows - input_rows)
    rows_odd = (padding_rows % 2 != 0)

    # same for padding_cols
    input_cols = x.size(3)
    effective_filter_size_cols = (kernel_size[1] - 1) * dilation[1] + 1
    out_cols = (input_cols + stride[1] - 1) // stride[1]
    padding_cols = max(0, (out_cols - 1) * stride[1] + effective_filter_size_cols - input_cols)
    cols_odd = (padding_cols % 2 != 0)
    x = F.pad(x, [(padding_cols // 2) - padding[0], (padding_cols // 2) + int(cols_odd) - padding[0],
                  (padding_rows // 2) - padding[1], (padding_rows // 2) + int(rows_odd) - padding[1]])
    return conv(x)


def upsample_cat(x_up, x_down):
    return torch.cat([x_up, x_down], dim=1)
