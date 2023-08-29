import sys
import torch
import torch.nn as nn
from utils import *


class Net(nn.Module):
    def __init__(self, inputsize, taskcla, use_sigmoid, nc=28):
        super().__init__()

        ncha, size, _ = inputsize
        self.taskcla = taskcla

        self.nLearner = 5
        self.nc = nc

        self.use_sigmoid = use_sigmoid
        self.s_gate = 100
        
        self.net1 = nn.Sequential(
            nn.Conv2d(ncha, self.nc, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(self.nc, self.nc, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(self.nc, self.nc, kernel_size=3),
            nn.Conv2d(self.nc, self.nc, kernel_size=3),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(16 * self.nc, 16 * self.nc)
        if self.use_sigmoid:
            self.efc1 = torch.nn.Embedding(len(self.taskcla), 16 * self.nc)

        self.net2 = nn.Sequential(
            nn.Conv2d(ncha, self.nc, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(self.nc, self.nc, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(self.nc, self.nc, kernel_size=3),
            nn.Conv2d(self.nc, self.nc, kernel_size=3),
            nn.MaxPool2d(2)
        )
        self.fc2 = nn.Linear(16 * self.nc, 16 * self.nc)
        if self.use_sigmoid:
            self.efc2 = torch.nn.Embedding(len(self.taskcla), 16 * self.nc)

        self.net3 = nn.Sequential(
            nn.Conv2d(ncha, self.nc, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(self.nc, self.nc, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(self.nc, self.nc, kernel_size=3),
            nn.Conv2d(self.nc, self.nc, kernel_size=3),
            nn.MaxPool2d(2)
        )
        self.fc3 = nn.Linear(16 * self.nc, 16 * self.nc)
        if self.use_sigmoid:
            self.efc3 = torch.nn.Embedding(len(self.taskcla), 16 * self.nc)

        self.net4 = nn.Sequential(
            nn.Conv2d(ncha, self.nc, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(self.nc, self.nc, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(self.nc, self.nc, kernel_size=3),
            nn.Conv2d(self.nc, self.nc, kernel_size=3),
            nn.MaxPool2d(2)
        )
        self.fc4 = nn.Linear(16 * self.nc, 16 * self.nc)
        if self.use_sigmoid:
            self.efc4 = torch.nn.Embedding(len(self.taskcla), 16 * self.nc)

        self.net5 = nn.Sequential(
            nn.Conv2d(ncha, self.nc, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(self.nc, self.nc, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(self.nc, self.nc, kernel_size=3),
            nn.Conv2d(self.nc, self.nc, kernel_size=3),
            nn.MaxPool2d(2)
        )
        self.fc5 = nn.Linear(16 * self.nc, 16 * self.nc)
        if self.use_sigmoid:
            self.efc5 = torch.nn.Embedding(len(self.taskcla), 16 * self.nc)

        self.last = torch.nn.ModuleList()

        for t, n in self.taskcla:
            self.last.append(torch.nn.Linear(4 * 4 * self.nc, n))  # 4*4*64 = 1024

        self.relu = torch.nn.ReLU()
        self.sig_gate = torch.nn.Sigmoid()

    def forward(self, x, t, return_expert=False, avg_act=False):
        if self.use_sigmoid:
            masks = self.mask(t, s=self.s_gate)
            gfc1, gfc2, gfc3, gfc4, gfc5 = masks

            self.Learners = []
            self.Learners_feature = []

            h1 = self.net1(x)
            h1 = h1.view(x.shape[0], -1)
            self.Learners_feature.append(h1)
            h1 = self.relu(self.fc1(h1))
            h1 = h1 * gfc1.expand_as(h1)
            self.Learners.append(h1.unsqueeze(0))

            h2 = self.net2(x)
            h2 = h2.view(x.shape[0], -1)
            self.Learners_feature.append(h2)
            h2 = self.relu(self.fc2(h2))
            h2 = h2 * gfc2.expand_as(h2)
            self.Learners.append(h2.unsqueeze(0))

            h3 = self.net3(x)
            h3 = h3.view(x.shape[0], -1)
            self.Learners_feature.append(h3)
            h3 = self.relu(self.fc3(h3))
            h3 = h3 * gfc3.expand_as(h3)
            self.Learners.append(h3.unsqueeze(0))

            h4 = self.net4(x)
            h4 = h4.view(x.shape[0], -1)
            self.Learners_feature.append(h4)
            h4 = self.relu(self.fc4(h4))
            h4 = h4 * gfc4.expand_as(h4)
            self.Learners.append(h4.unsqueeze(0))

            h5 = self.net5(x)
            h5 = h5.view(x.shape[0], -1)
            self.Learners_feature.append(h5)
            h5 = self.relu(self.fc5(h5))
            h5 = h5 * gfc5.expand_as(h5)
            self.Learners.append(h5.unsqueeze(0))

            h = torch.cat([h_result for h_result in self.Learners], 0)
            h = torch.sum(h, dim=0).squeeze(0)  # / self.nLearner

        else:
            self.Learners = []
            self.Learners_feature = []

            h1 = self.net1(x)
            h1 = h1.view(x.shape[0], -1)
            self.Learners_feature.append(h1)
            h1 = self.relu(self.fc1(h1))
            self.Learners.append(h1.unsqueeze(0))

            h2 = self.net2(x)
            h2 = h2.view(x.shape[0], -1)
            self.Learners_feature.append(h2)
            h2 = self.relu(self.fc2(h2))
            self.Learners.append(h2.unsqueeze(0))

            h3 = self.net3(x)
            h3 = h3.view(x.shape[0], -1)
            self.Learners_feature.append(h3)
            h3 = self.relu(self.fc3(h3))
            self.Learners.append(h3.unsqueeze(0))

            h4 = self.net4(x)
            h4 = h4.view(x.shape[0], -1)
            self.Learners_feature.append(h4)
            h4 = self.relu(self.fc4(h4))
            self.Learners.append(h4.unsqueeze(0))

            h5 = self.net5(x)
            h5 = h5.view(x.shape[0], -1)
            self.Learners_feature.append(h5)
            h5 = self.relu(self.fc5(h5))
            self.Learners.append(h5.unsqueeze(0))

            h = torch.cat([h_result for h_result in self.Learners], 0)
            h = torch.sum(h, dim=0).squeeze(0)  # / self.nLearner

        y = self.last[t](h)

        self.grads = {}

        def save_grad(name):
            def hook(grad):
                self.grads[name] = grad

            return hook

        if avg_act == True:
            names = [0, 1, 2, 3]
            act = [act1, act2, act3, act4]

            self.act = []
            for i in act:
                self.act.append(i.detach())
            for idx, name in enumerate(names):
                act[idx].register_hook(save_grad(name))

        if return_expert:
            self.Learners_y = []
            for i in range(self.nLearner):
                h_exp = self.Learners[i].squeeze(0)

                # using joint classifier
                y_exp = self.last[t](h_exp)
                self.Learners_y.append(y_exp)

            return y, self.Learners_y, self.Learners

        else:
            return y

    def mask(self,t,s=1):
        gfc1 = self.sig_gate(s * self.efc1(t))
        gfc2 = self.sig_gate(s * self.efc2(t))
        gfc3 = self.sig_gate(s * self.efc3(t))
        gfc4 = self.sig_gate(s * self.efc4(t))
        gfc5 = self.sig_gate(s * self.efc5(t))
        return [gfc1,gfc2,gfc3,gfc4,gfc5]
