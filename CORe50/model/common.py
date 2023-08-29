# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d, max_pool2d
import numpy as np
import pdb
from torch.nn.utils import weight_norm as wn
from itertools import chain


class AlexNet(nn.Module):

    def __init__(self, num_classes=50):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.dropout = nn.Dropout()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)

        self.linear = nn.Linear(4096, int(num_classes))

    def forward(self, x, avg_act=False):
        bsz = x.size(0)
        if x.dim() < 4:
            x = x.view(bsz,3,32,32)

        act1 = self.relu(self.conv1(x))
        x = self.maxpool(act1)
        act2 = self.relu(self.conv2(x))
        x = self.maxpool(act2)
        act3 = self.relu(self.conv3(x))
        act4 = self.relu(self.conv4(act3))
        act5 = self.relu(self.conv5(act4))
        x = self.maxpool(act5)
        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        act6 = self.relu(self.fc1(self.dropout(x)))
        act7 = self.relu(self.fc2(self.dropout(act6)))

        y = self.linear(act7)

        self.grads = {}
        self.act = []

        def save_grad(name):
            def hook(grad):
                self.grads[name] = grad

            return hook

        if avg_act == True:
            names = [0, 1, 2, 3, 4, 5, 6]
            act = [act1, act2, act3, act4, act5, act6, act7]

            self.act = []
            for i in act:
                self.act.append(i.detach())
            for idx, name in enumerate(names):
                act[idx].register_hook(save_grad(name))

        return y


class AlexNet_MCL(nn.Module):

    def __init__(self, num_classes=50):
        super(AlexNet_MCL, self).__init__()
        self.num_classes = num_classes

        self.nLearner = 5
        self.f_size = 32

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout = nn.Dropout()

        self.net1 = nn.Sequential(
            nn.Conv2d(3, self.f_size, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(self.f_size, self.f_size * 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(self.f_size * 3, self.f_size * 6, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.f_size * 6, self.f_size * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.f_size * 4, self.f_size * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.fc1 = nn.Linear(self.f_size * 64, self.f_size * 64)
        self.fc_1 = nn.Linear(self.f_size * 64, self.f_size * 64)

        self.net2 = nn.Sequential(
            nn.Conv2d(3, self.f_size, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(self.f_size, self.f_size * 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(self.f_size * 3, self.f_size * 6, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.f_size * 6, self.f_size * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.f_size * 4, self.f_size * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.fc2 = nn.Linear(self.f_size * 64, self.f_size * 64)
        self.fc_2 = nn.Linear(self.f_size * 64, self.f_size * 64)

        self.net3 = nn.Sequential(
            nn.Conv2d(3, self.f_size, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(self.f_size, self.f_size * 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(self.f_size * 3, self.f_size * 6, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.f_size * 6, self.f_size * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.f_size * 4, self.f_size * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.fc3 = nn.Linear(self.f_size * 64, self.f_size * 64)
        self.fc_3 = nn.Linear(self.f_size * 64, self.f_size * 64)

        self.net4 = nn.Sequential(
            nn.Conv2d(3, self.f_size, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(self.f_size, self.f_size * 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(self.f_size * 3, self.f_size * 6, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.f_size * 6, self.f_size * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.f_size * 4, self.f_size * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.fc4 = nn.Linear(self.f_size * 64, self.f_size * 64)
        self.fc_4 = nn.Linear(self.f_size * 64, self.f_size * 64)

        self.net5 = nn.Sequential(
            nn.Conv2d(3, self.f_size, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(self.f_size, self.f_size * 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(self.f_size * 3, self.f_size * 6, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.f_size * 6, self.f_size * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.f_size * 4, self.f_size * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.fc5 = nn.Linear(self.f_size * 64, self.f_size * 64)
        self.fc_5 = nn.Linear(self.f_size * 64, self.f_size * 64)

        self.linear = nn.Linear(self.f_size * 64, int(num_classes))
        self.adaptor = nn.Parameter(torch.ones((14, self.nLearner)))

    def forward(self, x, t=0, avg_act=False, return_expert=False):
        bsz = x.size(0)
        if x.dim() < 4:
            x = x.view(bsz,3,32,32)

        self.Experts = []

        h1 = self.net1(x)
        h1 = h1.view(x.shape[0], -1)
        h1 = self.relu(self.fc1(self.dropout(h1)))
        h1 = self.relu(self.fc_1(self.dropout(h1)))
        self.Experts.append(h1.unsqueeze(0))

        h2 = self.net2(x)
        h2 = h2.view(x.shape[0], -1)
        h2 = self.relu(self.fc2(self.dropout(h2)))
        h2 = self.relu(self.fc_2(self.dropout(h2)))
        self.Experts.append(h2.unsqueeze(0))

        h3 = self.net3(x)
        h3 = h3.view(x.shape[0], -1)
        h3 = self.relu(self.fc3(self.dropout(h3)))
        h3 = self.relu(self.fc_3(self.dropout(h3)))
        self.Experts.append(h3.unsqueeze(0))

        h4 = self.net4(x)
        h4 = h4.view(x.shape[0], -1)
        h4 = self.relu(self.fc4(self.dropout(h4)))
        h4 = self.relu(self.fc_4(self.dropout(h4)))
        self.Experts.append(h4.unsqueeze(0))

        h5 = self.net5(x)
        h5 = h5.view(x.shape[0], -1)
        h5 = self.relu(self.fc5(self.dropout(h5)))
        h5 = self.relu(self.fc_5(self.dropout(h5)))
        self.Experts.append(h5.unsqueeze(0))

        h = torch.cat([h_result for h_result in self.Experts], 0)
        h = torch.sum(h, dim=0).squeeze(0)

        y = self.linear(h)

        self.grads = {}
        self.act = []

        def save_grad(name):
            def hook(grad):
                self.grads[name] = grad

            return hook

        if avg_act == True:
            names = [0, 1, 2, 3, 4, 5, 6]
            act = [act1, act2, act3, act4, act5, act6, act7]

            self.act = []
            for i in act:
                self.act.append(i.detach())
            for idx, name in enumerate(names):
                act[idx].register_hook(save_grad(name))

        if return_expert:
            self.Experts_y = []
            for i in range(self.nLearner):
                h_exp = self.Experts[i].squeeze(0)

                # using joint classifier
                y_exp = self.linear(h_exp)
                self.Experts_y.append(y_exp)

            return y, self.Experts_y, self.Experts, self.Experts

        else:
            return y

