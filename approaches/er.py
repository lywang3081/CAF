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
    """ Class implementing the fine tuning """
    def __init__(self,model,nepochs=100,sbatch=256,lr=0.001,lr_min=1e-6,lr_factor=3,lr_patience=5,clipgrad=100,args=None,log_name = None):
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
        self.lamb=args.lamb
        if len(args.parameter)>=1:
            params=args.parameter.split(',')
            print('Setting parameters to',params)
            self.lamb=float(params[0])

        self.xtrain_mb = []
        self.ytrain_mb = []
        self.mb_size = 20 # num of images per class
        self.fixed_mb_size = 2000 #20x100

        return

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        return optimizer

    def train(self, t, xtrain, ytrain, xvalid, yvalid, data, input_size, taskcla):
        best_loss = np.inf
        best_model = utils.get_model(self.model)
        lr = self.lr
        self.optimizer = self._get_optimizer(lr)

        #if t > 0:
        #    xtrain_mb_all = torch.cat(self.xtrain_mb, 0)
        #    ytrain_mb_all = torch.cat(self.ytrain_mb, 0)

        if t > 0:
            self.xtrain_mb_all = []
            self.ytrain_mb_all = []
            for t_old in range(t):
                self.xtrain_mb_all.append([])
                self.ytrain_mb_all.append([])
                for i in range(len(self.ytrain_mb[t_old])):
                    if i == 0:
                        self.xtrain_mb_all[t_old] = self.xtrain_mb[t_old][i]
                        self.ytrain_mb_all[t_old] = self.ytrain_mb[t_old][i]
                    else:
                        self.xtrain_mb_all[t_old] = torch.cat((self.xtrain_mb_all[t_old], self.xtrain_mb[t_old][i]))
                        self.ytrain_mb_all[t_old] = torch.cat((self.ytrain_mb_all[t_old], self.ytrain_mb[t_old][i]))

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0=time.time()

            num_batch = xtrain.size(0)
            self.train_epoch(t, xtrain, ytrain, e)

            '''
            if t == 0:
                num_batch = xtrain.size(0)
                self.train_epoch(t, xtrain, ytrain, e)
            else:

                for i in range(len(self.ytrain_mb)):
                    if i == 0:
                        xtrain_all = torch.cat((xtrain, self.xtrain_mb[i]))
                        ytrain_all = torch.cat((ytrain, self.ytrain_mb[i]))
                    else:
                        xtrain_all = torch.cat((xtrain_all, self.xtrain_mb[i]))
                        ytrain_all = torch.cat((ytrain_all, self.ytrain_mb[i]))

                num_batch = xtrain.size(0)
                self.train_epoch(t, xtrain, xtrain, e)
            '''

            clock1=time.time()
            train_loss,train_acc=self.eval(t,xtrain,ytrain)
            clock2=time.time()
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(
                e+1,1000*self.sbatch*(clock1-clock0)/num_batch, 1000*self.sbatch*(clock2-clock1)/num_batch,train_loss,100*train_acc),end='')
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


        cls_unique = torch.unique(ytrain)
        self.xtrain_mb.append([])
        self.ytrain_mb.append([])

        '''
        #fixed memory budget
        self.mb_size = self.fixed_mb_size // (len(cls_unique) * (t+1))
        print("Current Memory Size:", self.mb_size)
        if t > 0:
            for i in range(len(self.xtrain_mb)):
                self.xtrain_mb[i] = self.xtrain_mb[i][:self.mb_size]
                self.ytrain_mb[i] = self.ytrain_mb[i][:self.mb_size]
        '''

        for i in range(len(cls_unique)):
            cls = np.array(cls_unique.cpu())[i]
            idx = np.where(np.array(ytrain.cpu())==cls)
            idx_mb = random.sample(list(idx[0]), self.mb_size)
            mb_x = xtrain[idx_mb]
            mb_y = ytrain[idx_mb]
            self.xtrain_mb[t].append(mb_x)
            self.ytrain_mb[t].append(mb_y)


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
            outputs = self.model.forward(images)[t]
            loss=self.criterion(t,outputs,targets)

            if t > 0:
                for t_old in range(t):
                    idx_er = np.arange(len(self.ytrain_mb_all[t_old]))
                    #print("idx_er", idx_er)
                    np.random.shuffle(idx_er)
                    idx_er = idx_er[:64]
                    #print("idx_er", idx_er)
                    x_er = self.xtrain_mb_all[t_old][idx_er]
                    y_er = self.ytrain_mb_all[t_old][idx_er]
                    #print("y_er", y_er)
                    outputs_ = self.model.forward(x_er)[t_old]
                    loss += self.criterion(t_old, outputs_, y_er)

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
            
            output = self.model.forward(images)[t]
            
            loss=self.criterion(t,output,targets)
            _,pred=output.max(1)
            hits=(pred==targets).float()

            # Log
            total_loss+=loss.data.cpu().numpy()*len(b)
            total_acc+=hits.sum().data.cpu().numpy()
            total_num+=len(b)

        return total_loss/total_num,total_acc/total_num

    def criterion(self,t,output,targets):
        return self.ce(output,targets)