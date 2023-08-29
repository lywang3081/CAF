import sys,time,os
import numpy as np
import torch
from copy import deepcopy
import utils
from utils import *
sys.path.append('..')
from arguments import get_args
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
args = get_args()

class Appr(object):
    def __init__(self,model,nepochs=100,sbatch=256,lr=0.001,lr_min=1e-6,lr_factor=3,lr_patience=5,clipgrad=100,args=None,log_name = None, use_sigmoid=False):
        self.model=model
        self.model_old=model
        self.fisher=None

        self.nepochs = nepochs
        self.sbatch = sbatch
        self.lr = lr
        self.lr_min = lr_min * 1/3
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad

        self.lamb = args.lamb
        self.lamb_kld = args.lamb_kld
        self.lamb_af = args.lamb_af

        self.use_sigmoid = use_sigmoid
        self.model.s_gate = args.s_gate

        self.ce=torch.nn.CrossEntropyLoss()
        self.optimizer=self._get_optimizer()
        
        self.omega = {}
        for n,_ in self.model.named_parameters():
            self.omega[n] = 0

        if 'cifar' in args.approach:
            self.kld = KLD_adapt()
        else:
            self.kld = KLD()

        self.mcl = args.mcl
        self.adapt_af = True
        self.adapt_kld = True

        return

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        return optimizer

    def train(self, t, xtrain, ytrain, xvalid, yvalid, data, input_size, taskcla):

        #initialization of adaptor
        self.model.adaptor = nn.init.normal_(nn.Parameter(torch.ones((14, self.model.nLearner))))
        self.model.adaptor_kld = nn.init.normal_(nn.Parameter(torch.ones((self.model.nLearner, self.model.nLearner))))

        # if using the same initialization for each learner
        if self.mcl != 'mcl-h':
        #if self.same_init:
            if t == 0:
                print("Same Random Initialization")
                weight_id = 0
                weight_num = 14
                init_exp = []
                for n, p in self.model.named_parameters():
                    if 'last' not in n and 'adaptor' not in n:
                        if weight_id < weight_num:
                            init_exp.append(p.data.clone().detach())
                        else:
                            p.data = init_exp[weight_id % weight_num]
                        weight_id += 1

        best_loss = np.inf
        best_model = utils.get_model(self.model)
        lr = self.lr
        self.optimizer = self._get_optimizer(lr)
        
        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0=time.time()
            num_batch = xtrain.size(0)
            
            self.train_epoch(t,xtrain,ytrain)
            
            clock1=time.time()
            train_loss,train_acc=self.eval(t,xtrain,ytrain)
            clock2=time.time()
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(
                e+1,1000*self.sbatch*(clock1-clock0)/num_batch,
                1000*self.sbatch*(clock2-clock1)/num_batch,train_loss,100*train_acc),end='')
            # Valid
            valid_loss,valid_acc=self.eval(t,xvalid,yvalid)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
            print(' lr : {:.6f}'.format(self.optimizer.param_groups[0]['lr']))
            #save log for current task & old tasks at every epoch

            # Adapt lr
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = utils.get_model(self.model)
                patience = self.lr_patience
                print(' *', end='')

            else:
                patience -= 1
                if patience <= 0:
                    lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    if lr < self.lr_min:
                        print()
                    patience = self.lr_patience
                    self.optimizer = self._get_optimizer(lr)
            print()

        # Restore best
        utils.set_model_(self.model, best_model)

        if self.use_sigmoid:
            task = torch.autograd.Variable(torch.LongTensor([t]).cuda())
            mask = self.model.mask(task, s=self.model.s_gate)
            for i in range(len(mask)):
                mask[i] = torch.autograd.Variable(mask[i].data.clone(), requires_grad=False)

        self.old_param = {}
        for n, p in self.model.named_parameters():
            self.old_param[n] = p.data.clone().detach()

        self.omega_update(t,xtrain)
        
        return

    def train_epoch(self,t,x,y):
        self.model.train()

        r=np.arange(x.size(0))
        np.random.shuffle(r)
        r=torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images=x[b]
            targets=y[b]
            
            # Forward current model
            task = torch.autograd.Variable(torch.LongTensor([t]).cuda())
            outputs, outputs_expert, feature_expert = self.model.forward(images, task, return_expert=True)

            loss = self.criterion(t, outputs, targets) + self.lamb_kld * self.kld(outputs_expert, t, self.model)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return

    def eval(self,t,x,y):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()

        r = np.arange(x.size(0))
        r = torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images=x[b]
            targets=y[b]

            # Forward
            task = torch.autograd.Variable(torch.LongTensor([t]).cuda())
            outputs = self.model.forward(images, task)

            loss = self.criterion(t,outputs,targets)

            _,pred=outputs.max(1)
            hits=(pred==targets).float()

            # Log
            total_loss+=loss.data.cpu().numpy()*len(b)
            total_acc+=hits.sum().data.cpu().numpy()
            total_num+=len(b)

        return total_loss/total_num,total_acc/total_num

    def criterion(self, t, output, targets):
        # Regularization for all previous tasks
        loss_reg = 0
        loss_af = 0
        weight_id = 0
        weight_num = 14

        if t > 0:
            for name, param in self.model.named_parameters():
                if 'last' not in name and 'adaptor' not in name:
                    loss_reg += torch.sum(self.omega[name] * (self.old_param[name] - param).pow(2)) / 2
                if 'efc' not in name:
                    if self.adapt_af:
                        if 'last' not in name and 'adaptor' not in name:
                            softmax_adaptor = self.model.nLearner * F.softmax(self.model.adaptor[weight_id % weight_num])
                            loss_af += softmax_adaptor[weight_id // weight_num] * torch.sum(param.pow(2)) / 2
                            weight_id += 1
                        elif 'last' in name: #shared output head
                            loss_af += torch.sum(param.pow(2)) / 2
                    else:
                        loss_af += torch.sum(param.pow(2)) / 2

        return self.ce(output, targets) + self.lamb * loss_reg + self.lamb_af * loss_af
    
    def omega_update(self,t,x):
        sbatch = 20
        
        # Compute
        self.model.train()
        for i in tqdm(range(0,x.size(0),sbatch),desc='Omega',ncols=100,ascii=True):
            b=torch.LongTensor(np.arange(i,np.min([i+sbatch,x.size(0)]))).cuda()
            images = x[b]
            # Forward and backward
            self.model.zero_grad()

            task = torch.autograd.Variable(torch.LongTensor([t]).cuda())
            # Forward
            outputs = self.model.forward(images, task)

            # Sum of L2 norm of output scores
            loss = torch.sum(outputs.norm(2, dim = -1))

            loss.backward()

            # Get gradients
            for n,p in self.model.named_parameters():
                if p.grad is not None and 'adaptor' not in n:
                    self.omega[n]+= p.grad.data.abs() / x.size(0)

        return