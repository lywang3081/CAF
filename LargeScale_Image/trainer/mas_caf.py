from __future__ import print_function

import copy
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as td
from PIL import Image
from tqdm import tqdm
import trainer

import networks


class Trainer(trainer.GenericTrainer):
    def __init__(self, model, args, optimizer, evaluator, taskcla):
        super().__init__(model, args, optimizer, evaluator, taskcla)
        
        self.lamb=args.lamb
        model.s_gate = args.s_gate

        self.lamb_kld = args.lamb_kld
        self.lamb_af = args.lamb_af

        self.adapt_af = True
        self.adapt_kld = True

        self.kld = KLD()

        self.omega = {}
        for n,_ in self.model.named_parameters():
            self.omega[n] = 0

    def update_lr(self, epoch, schedule):
        for temp in range(0, len(schedule)):
            if schedule[temp] == epoch:
                for param_group in self.optimizer.param_groups:
                    self.current_lr = param_group['lr']
                    param_group['lr'] = self.current_lr * self.args.gammas[temp]
                    print("Changing learning rate from %0.4f to %0.4f"%(self.current_lr, self.current_lr * self.args.gammas[temp]))
                    self.current_lr *= self.args.gammas[temp]

    def setup_training(self, lr):
        
        for param_group in self.optimizer.param_groups:
            print("Setting LR to %0.4f"%lr)
            param_group['lr'] = lr
            self.current_lr = lr

    def update_frozen_model(self):
        self.model.eval()
        self.model_fixed = copy.deepcopy(self.model)
        self.model_fixed.eval()
        for param in self.model_fixed.parameters():
            param.requires_grad = False

    def train(self, train_loader, test_loader, t):
        #initialization of adaptor
        self.model.adaptor = nn.init.normal_(nn.Parameter(torch.ones((14, self.model.nLearner))))
        self.model.adaptor_kld = nn.init.normal_(nn.Parameter(torch.ones((self.model.nLearner, self.model.nLearner))))

        lr = self.args.lr
        self.setup_training(lr)
        # Do not update self.t
        if t>0:
            #self.update_frozen_model()
            self.old_param = {}
            for n, p in self.model.named_parameters():
                self.old_param[n] = p.data.clone().detach()

            self.omega_update()
        
        # Now, you can update self.t
        self.t = t
        kwargs = {'num_workers': 8, 'pin_memory': True}
        self.train_iterator = torch.utils.data.DataLoader(train_loader, batch_size=self.args.batch_size, shuffle=True, **kwargs)
        self.test_iterator = torch.utils.data.DataLoader(test_loader, 100, shuffle=False, **kwargs)
        self.omega_iterator = torch.utils.data.DataLoader(train_loader, batch_size=20, shuffle=True, **kwargs)
        for epoch in range(self.args.nepochs):
            self.model.train()
            self.update_lr(epoch, self.args.schedule)
            for samples in tqdm(self.train_iterator):
                data, target = samples
                data, target = data.cuda(), target.cuda()

                #output = self.model(data)[t]
                task = torch.autograd.Variable(torch.LongTensor([self.t]).cuda())
                output, outputs_expert, _ = self.model.forward(data, task, return_expert=True)#[t] #(data, t)
                output = output[t]
                loss_CE = self.criterion(output,target) + self.lamb_kld * self.kld(outputs_expert)

                self.optimizer.zero_grad()
                (loss_CE).backward()
                self.optimizer.step()

            train_loss,train_acc = self.evaluator.evaluate(self.model, self.train_iterator, t)
            num_batch = len(self.train_iterator)
            print('| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% |'.format(epoch+1,train_loss,100*train_acc),end='')
            valid_loss,valid_acc=self.evaluator.evaluate(self.model, self.test_iterator, t)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
            print()
    
    def criterion(self, output, targets):
        # Regularization for all previous tasks
        loss_reg = 0
        loss_af = 0
        weight_id = 0
        weight_num = 12

        if self.t > 0:
            for name, param in self.model.named_parameters():
                if 'adaptor' not in name:
                    loss_reg += torch.sum(self.omega[name].cuda() * (self.old_param[name].cuda() - param.cuda()).pow(2)) / 2
                if self.adapt_af:
                    if 'last' not in name and 'adaptor' not in name:
                        softmax_adaptor = self.model.nLearner * F.softmax(self.model.adaptor[weight_id % weight_num])
                        loss_af += softmax_adaptor[weight_id // weight_num] * torch.sum(param.pow(2)) / 2

                        weight_id += 1
                    elif 'last' in name:
                        loss_af += torch.sum(param.pow(2)) / 2
                else:
                    loss_af += torch.sum(param.cuda().pow(2)) / 2

        else:
            for name, param in self.model.named_parameters():
                if self.adapt_af:
                    if 'last' not in name and 'adaptor' not in name:
                        softmax_adaptor = self.model.nLearner * F.softmax(self.model.adaptor[weight_id % weight_num])
                        loss_af += softmax_adaptor[weight_id // weight_num] * torch.sum(param.pow(2)) / 2

                        weight_id += 1
                    elif 'last' in name:
                        loss_af += torch.sum(param.pow(2)) / 2
                else:
                    loss_af += torch.sum(param.cuda().pow(2)) / 2

        return self.ce(output, targets) + self.lamb * loss_reg + self.lamb_af * loss_af 
        
    
    def omega_update(self):
        sbatch = 20
        
        # Compute
        self.model.train()
        for samples in tqdm(self.omega_iterator):
            data, target = samples
            data, target = data.cuda(), target.cuda()
            # Forward and backward
            self.model.zero_grad()

            #outputs = self.model.forward(data)[self.t]
            task = torch.autograd.Variable(torch.LongTensor([self.t]).cuda())
            outputs = self.model.forward(data, task)[self.t]

            # Sum of L2 norm of output scores
            loss = torch.sum(outputs.norm(2, dim = -1))
            loss.backward()

            # Get gradients
            for n,p in self.model.named_parameters():
                if p.grad is not None:
                    self.omega[n]+= p.grad.data.abs() / len(self.train_iterator)

        return 


class KLD(nn.Module):
    def __init__(self):
        super(KLD, self).__init__()
        self.criterion_KLD = nn.KLDivLoss(reduction='batchmean')

    def forward(self, x):
        KLD_loss = 0
        for k in range(len(x)):
            for l in range(len(x)):
                if l != k:
                    KLD_loss += self.criterion_KLD(F.log_softmax(x[k], dim=1), F.softmax(x[l], dim=1).detach())

        return KLD_loss
