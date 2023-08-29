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
args = get_args()


class Appr(object):
    """ Class implementing the Elastic Weight Consolidation approach described in http://arxiv.org/abs/1612.00796 """
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

        self.ce=torch.nn.CrossEntropyLoss()
        self.optimizer=self._get_optimizer()

        self.lamb = args.lamb
        self.lamb_kld = args.lamb_kld
        self.lamb_af = args.lamb_af

        self.use_sigmoid = use_sigmoid
        self.model.s_gate = args.s_gate

        if len(args.parameter)>=1:
            params=args.parameter.split(',')
            print('Setting parameters to',params)
            self.lamb=float(params[0])

        if 'cifar' in args.experiment:
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

            self.train_epoch(t,xtrain,ytrain, e)

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

        # Fisher ops
        if t>0:
            fisher_old={}
            for n,_ in self.model.named_parameters():
                fisher_old[n]=self.fisher[n].clone()
        self.fisher=utils.fisher_matrix_diag_coscl(t,xtrain,ytrain,self.model,self.criterion)
        if t>0:
            # Watch out! We do not want to keep t models (or fisher diagonals) in memory, therefore we have to merge fisher diagonals
            for n,_ in self.model.named_parameters():
                self.fisher[n] = (self.fisher[n] + fisher_old[n] * t) / (t + 1)
                '''
                if 's_cifar100' or 'omniglot' in args.experiment:
                    self.fisher[n] = self.fisher[n] + fisher_old[n] #slightly better performance for MCL rather than SCL
                else:
                    self.fisher[n]=(self.fisher[n]+fisher_old[n]*t)/(t+1)
                '''

        return

    def train_epoch(self,t,x,y, epoch):
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
            outputs, outputs_expert, logit_expert = self.model.forward(images,task, return_expert=True)

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
            output = self.model.forward(images, task)

            loss=self.criterion(t,output,targets)
            _,pred=output.max(1)
            hits=(pred==targets).float()

            # Log
            total_loss+=loss.data.cpu().numpy()*len(b)
            total_acc+=hits.sum().data.cpu().numpy()
            total_num+=len(b)

        return total_loss/total_num,total_acc/total_num

    def eval_expert(self,t,x,y):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()

        r = np.arange(x.size(0))
        r = torch.LongTensor(r).cuda()

        total_hits_exp = []
        for j in range(self.model.nLearner):
            total_hits_exp.append(0)

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images=x[b]
            targets=y[b]
            task = torch.autograd.Variable(torch.LongTensor([t]).cuda())
            # Forward
            output, output_exp, logit_exp = self.model.forward(images, task, return_expert=True)

            loss=self.criterion(t,output,targets)
            _,pred=output.max(1)
            hits=(pred==targets).float()

            # Log
            total_loss+=loss.data.cpu().numpy()*len(b)
            total_acc+=hits.sum().data.cpu().numpy()
            total_num+=len(b)
            #print("total_acc", total_acc)

            for j in range(self.model.nLearner):
                _, pred = output_exp[j].max(1)
                hits_exp = (pred == targets).float()
                total_hits_exp[j] += hits_exp.sum().data.cpu().numpy()

        for j in range(self.model.nLearner):
            total_hits_exp[j] = total_hits_exp[j]/total_num

        return total_loss/total_num, total_acc/total_num, total_hits_exp

    def criterion(self, t, output, targets):
        # Regularization for all previous tasks
        loss_reg = 0
        loss_af = 0
        weight_id = 0
        weight_num = 14

        if t > 0:
            for name, param in self.model.named_parameters():
                if 'last' not in name and 'adaptor' not in name:
                    loss_reg += torch.sum(self.fisher[name] * (self.old_param[name] - param).pow(2)) / 2
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


