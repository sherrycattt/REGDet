import torch
from torch import nn

from .blocks import get_attention_layer, get_conv_rnn_layer


class SCAN(nn.Module):

    def __init__(self, input_nc=3, output_nc=3, n_filters=24, n_blocks=7, reduction=16, unit='Conv',
                 attention_type='eca', attention_last=None, use_dilation=True, res_mode='None'):
        super(SCAN, self).__init__()
        self.res_mode = res_mode
        self.model_list = nn.ModuleList(self.make_layers(
            input_nc=input_nc, n_filters=n_filters, n_blocks=(n_blocks - 2),
            unit=unit, attention_type=attention_type, reduction=reduction, use_dilation=use_dilation
        ))
        attention_last = attention_last or attention_type
        self.dec = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, 3, padding=1),
            get_attention_layer(attention_last)(n_filters, reduction),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_filters, output_nc, 1),
        )

    @staticmethod
    def make_layers(input_nc=3, n_filters=24, n_blocks=7, unit='Conv', attention_type='eca',
                    reduction=16, use_dilation=True):
        RecUnit = get_conv_rnn_layer(unit)
        if use_dilation:
            layers = [RecUnit(input_nc=input_nc, output_nc=n_filters, kernel_size=3, dilation=1,
                              attention_type=attention_type, reduction=reduction)] + \
                     [RecUnit(input_nc=n_filters, output_nc=n_filters, kernel_size=3, dilation=2 ** i,
                              attention_type=attention_type, reduction=reduction) for i in range(n_blocks - 1)]
        else:
            layers = [RecUnit(input_nc=input_nc, output_nc=n_filters, kernel_size=3, dilation=1,
                              attention_type=attention_type, reduction=reduction)] + \
                     [RecUnit(input_nc=n_filters, output_nc=n_filters, kernel_size=3, dilation=1,
                              attention_type=attention_type, reduction=reduction) for i in range(n_blocks - 1)]
        return layers

    def forward(self, x):
        ori = x
        for rnn in self.model_list:
            x, _ = rnn(x, None)
        x = self.dec(x)
        if self.res_mode.lower() == 'minus':
            x = ori - x
        elif self.res_mode.lower() == 'add':
            x = ori + x
        elif self.res_mode.lower() == 'mul':
            x = ori * x
        return x


class ConvRNN(SCAN):
    def __init__(self, input_nc=3, output_nc=3, n_filters=24, n_blocks=7, reduction=16,
                 unit='GRU', attention_type='eca', attention_last=None, use_dilation=True,
                 res_mode='None', stage_num=4, frame='Full'):
        super(ConvRNN, self).__init__(
            input_nc=input_nc, output_nc=output_nc, n_filters=n_filters, reduction=reduction, unit=unit,
            res_mode=res_mode, n_blocks=n_blocks,
            attention_type=attention_type, attention_last=attention_last, use_dilation=use_dilation
        )
        self.stage_num = stage_num
        self.frame = frame
        self.rnn_type = unit

    def forward(self, x):
        old_states = [None for _ in range(len(self.model_list))]  # n_blocks-2
        oups = []
        for i in range(self.stage_num):
            ori_x = x
            states = []
            for rnn, state in zip(self.model_list, old_states):
                x, st = rnn(x, state)
                states.append(st)
            old_states = states.copy()
            x = self.dec(x)

            if self.res_mode.lower() == 'minus':
                x = ori_x - x
            elif self.res_mode.lower() == 'add':
                x = ori_x + x
            elif self.res_mode.lower() == 'mul':
                x = ori_x * x
            oups.append(x)
        if self.rnn_type.lower() == 'conv':
            oups = oups[0]
            return oups
        else:
            oups = torch.cat(oups, dim=1)

        oups = 255 * (oups - 0.5)

        return oups
