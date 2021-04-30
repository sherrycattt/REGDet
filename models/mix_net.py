# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn


class MixNet(nn.Module):
    def __init__(self, detect_net, brighten_net, phase='test'):
        super(MixNet, self).__init__()
        self.brighten_net = brighten_net
        self.detect_net = detect_net
        self.phase = phase

    def forward(self, x):
        brighten_out = self.brighten_net(x)

        if self.phase == 'train':
            return brighten_out, self.detect_net(brighten_out)
        elif self.phase == 'test':
            return self.detect_net(brighten_out)
