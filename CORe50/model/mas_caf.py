# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from .common import AlexNet_MCL
import pdb
import torch.nn as nn
import torch.nn.functional as F


class Net(torch.nn.Module):

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        
        self.reg = args.lamb
        self.lamb1 = args.lamb1
        self.lamb2 = args.lamb2

        self.kld = KLD()

        # setup network
        self.is_cifar = any(x in str(args.data_file) for x in ['cifar', 'cub', 'core', 'mini'])
        self.net = AlexNet_MCL(n_outputs)

        # setup optimizer
        self.opt = torch.optim.SGD(self.net.parameters(), lr=args.lr)

        # setup losses
        self.bce = torch.nn.CrossEntropyLoss()
        self.criterion_KLD = nn.KLDivLoss(reduction='batchmean')

        # setup memories
        self.current_task = 0
        self.fisher = {}
        self.old_param = {}
        self.optpar = {}
        self.memx = None
        self.memy = None

        if self.is_cifar:
            self.nc_per_task = n_outputs / n_tasks
        else:
            self.nc_per_task = n_outputs
        self.n_outputs = n_outputs
        self.n_memories = args.n_memories

    def compute_offsets(self, task):
        if self.is_cifar:
            offset1 = task * self.nc_per_task
            offset2 = (task + 1) * self.nc_per_task
        else:
            offset1 = 0
            offset2 = self.n_outputs
        return int(offset1), int(offset2)

    def forward(self, x, t):
        output = self.net(x)
        if self.is_cifar:
            # make sure we predict classes within the current task
            offset1, offset2 = self.compute_offsets(t)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, int(offset2):self.n_outputs].data.fill_(-10e10)
        return output
    def on_epoch_end(self):
        pass

    def observe(self, x, t, y):
        self.net.train()

        if self.memx is None:
            self.memx = x.data.clone()
            self.memy = y.data.clone()
        else:
            if self.memx.size(0) < self.n_memories:
                self.memx = torch.cat((self.memx, x.data.clone()))
                self.memy = torch.cat((self.memy, y.data.clone()))
                if self.memx.size(0) > self.n_memories:
                    self.memx = self.memx[:self.n_memories]
                    self.memy = self.memy[:self.n_memories]

        self.net.zero_grad()
        if self.is_cifar:
            offset1, offset2 = self.compute_offsets(t)
            outputs, outputs_expert, logit_expert, feature_expert = self.net(x, return_expert=True)
            loss_afs = self.lamb1 * self.kld(outputs_expert, offset1, offset2)
            loss = self.bce((outputs[:, offset1: offset2]), y - offset1) + loss_afs
        else:
            loss = self.bce(self(x, t), y)

        # add ewc penalty
        loss_reg = 0
        loss_af = 0
        weight_id = 0
        weight_num = 14
        if t>0:
            for name, param in self.net.named_parameters():
                if 'linear' not in name:
                    loss_reg += torch.sum(self.fisher[name]*(self.old_param[name] - param).pow(2))/2
                    if self.adapt_af:
                        if 'linear' not in name and 'adaptor' not in name:
                            softmax_adaptor = self.net.nLearner * F.softmax(self.net.adaptor[weight_id % weight_num])
                            loss_af += softmax_adaptor[weight_id // weight_num] * torch.sum(param.pow(2)) / 2
                            weight_id += 1
                    else:
                        loss_af += torch.sum(param.pow(2)) / 2

        loss += self.reg * loss_reg + self.lamb2 * loss_af

        loss.backward()
        self.opt.step()
        return loss.item()

    def observe_fisher(self, x, t, y, fisher_curr):
        self.net.eval()
        self.net.zero_grad()
        offset1, offset2 = self.compute_offsets(t)
        loss = torch.sum((self.net(x)[:, offset1: offset2]).norm(2, dim=-1))
        loss.backward()

        for n,p in self.net.named_parameters():
            if p.grad is not None:
                fisher_curr[n]+= x.size(0) * p.grad.data.pow(2)

        self.net.zero_grad()
        self.net.train()
        return loss.item(), fisher_curr

class KLD(nn.Module):
    def __init__(self):
        super(KLD, self).__init__()
        self.criterion_KLD = nn.KLDivLoss(reduction='batchmean')

    def forward(self, x, offset1, offset2):
        KLD_loss = 0
        for k in range(len(x)):
            for l in range(len(x)):
                if l != k:
                    KLD_loss += self.criterion_KLD(F.log_softmax((x[k][:, offset1: offset2]), dim=1), F.softmax((x[l][:, offset1: offset2]), dim=1).detach())

        return KLD_loss