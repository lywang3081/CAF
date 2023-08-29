import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from tqdm import tqdm
from copy import deepcopy

from rl_module.ppo_model import Policy

class PPO_GS():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 optimizer,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 lamb = 100,
                 mu = 1,
                 eta = 0.9,
                 gamma = 1.0,
                ):

        self.actor_critic = actor_critic
        self.actor_critic_old = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        
        self.lamb=lamb 
        self.mu = mu
        self.eta=eta 
        self.gamma = gamma
        self.freeze = {}
        
        self.mask = {}
        
        for (name,module) in self.actor_critic.base.named_children():
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.Conv2d):
                continue
            bias = module.bias
            self.mask[name] = torch.zeros_like(bias)
        
        print ('lamb : ', self.lamb)
        print ('mu : ', self.mu)
        print ('eta : ', self.eta)
        print ('gamma : ', self.gamma)

        if optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.actor_critic.parameters(),lr=lr, momentum=0.9)
        elif optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr, eps=eps)

    def update(self, rollouts, task_num):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch, task_num)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                
                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()
                
                ## freeze operation
                if task_num>0:
                    for name, param in self.actor_critic_old.base.named_parameters():
                        if 'bias' in name or 'last' in name or 'conv1' in name:
                            continue
                        key = name.split('.')[0]
                        param.data = param.data*self.freeze[key]
                

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                
            self.proxy_grad_descent(task_num,lr)

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        
        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
    
    def proxy_grad_descent(self, t, lr):
        with torch.no_grad():
            for (name,module),(_,module_old) in zip(self.actor_critic.base.named_children(),self.actor_critic_old.base.named_children()):
                if not isinstance(module, torch.nn.Linear) and not isinstance(module, torch.nn.Conv2d):
                    continue

                mu = self.mu
                
                key = name
                weight = module.weight
                bias = module.bias
                weight_old = module_old.weight
                bias_old = module_old.bias
                
                if len(weight.size()) > 2:
                    norm = weight.norm(2, dim=(1,2,3))
                else:
                    norm = weight.norm(2, dim=(1))
                norm = (norm**2 + bias**2).pow(1/2)                

                aux = F.threshold(norm - mu * lr, 0, 0, False)
                alpha = aux/(aux+mu*lr)
                coeff = alpha * (1-self.mask[key])

                if len(weight.size()) > 2:
                    sparse_weight = weight.data * coeff.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) 
                else:
                    sparse_weight = weight.data * coeff.unsqueeze(-1) 
                sparse_bias = bias.data * coeff

                penalty_weight = 0
                penalty_bias = 0

                if t>0:
                    if len(weight.size()) > 2:
                        norm = (weight - weight_old).norm(2, dim=(1,2,3))
                    else:
                        norm = (weight - weight_old).norm(2, dim=(1))

                    norm = (norm**2 + (bias-bias_old)**2).pow(1/2)
                    
                    aux = F.threshold(norm - self.omega[key]*self.lamb*lr, 0, 0, False)
                    boonmo = lr*self.lamb*self.omega[key] + aux
                    alpha = (aux / boonmo)
                    alpha[alpha!=alpha] = 1
                        
                    coeff_alpha = alpha * self.mask[key]
                    coeff_beta = (1-alpha) * self.mask[key]


                    if len(weight.size()) > 2:
                        penalty_weight = coeff_alpha.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*weight.data + \
                                            coeff_beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*weight_old.data
                    else:
                        penalty_weight = coeff_alpha.unsqueeze(-1)*weight.data + coeff_beta.unsqueeze(-1)*weight_old.data
                    penalty_bias = coeff_alpha*bias.data + coeff_beta*bias_old.data

                weight.data = sparse_weight + penalty_weight
                bias.data = sparse_bias + penalty_bias
            
        return
    
    def freeze_model(self,model):
        for param in model.parameters():
            param.requires_grad = False
        return
    
    def freeze_init(self, task_idx):
        
        if task_idx>0:
            self.freeze = {}
            for name, param in self.actor_critic.base.named_parameters():
                if 'bias' in name or 'last' in name:
                    continue
                key = name.split('.')[0]
                if 'conv1' not in name:
                    if 'conv' in name: #convolution layer
                        temp = torch.ones_like(param)
                        temp[:, self.omega[prekey] == 0] = 0
                        temp[self.omega[key] == 0] = 1
                        self.freeze[key] = temp
                    else:#linear layer
                        temp = torch.ones_like(param)
                        temp = temp.reshape((temp.size(0), self.omega[prekey].size(0) , -1))
                        temp[:, self.omega[prekey] == 0] = 0
                        temp[self.omega[key] == 0] = 1
                        self.freeze[key] = temp.reshape(param.shape)
                prekey = key
        
    
    def gs_cal_rl(self,task_idx, rollouts, args, sbatch=20):
    
        # Init
        param_R = {}

        for name, param in self.actor_critic.base.named_parameters():
            if len(param.size()) == 1:
                continue
            name = name.split('.')[0]
            param = param.view(param.size(0), -1)
            param_R['{}'.format(name)]=torch.zeros((param.size(0)))

        # Compute
        self.actor_critic.train()
        
        for batch in tqdm(range(args.gs_epochs)):
            for step in range(args.num_gs_steps):
                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = self.actor_critic.act(
                        rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step], task_idx, True, True)

                cnt = 0

                for idx, j in enumerate(self.actor_critic.base.activations):
        #            j = j*model.grads[idx]
                    j = torch.mean(j, dim=0)
                    if len(j.size())>1:
                        j = torch.mean(j.view(j.size(0), -1), dim = 1).abs()
                    self.actor_critic.base.activations[idx] = j

                for name, module in self.actor_critic.base.named_children():
                    if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
                        param_R[name] += self.actor_critic.base.activations[cnt].abs().detach()*sbatch
                        cnt+=1 

            with torch.no_grad():
                for key in param_R.keys():
                    param_R[key]=(param_R[key]/args.num_gs_steps)

        return param_R
    
    def update_omega(self, rollouts, task_idx, taskcla, obs_shape, device, args):
    
        # Update old
        self.actor_critic.base.activation = None

        # omega ops
        temp=self.gs_cal_rl(task_idx,rollouts, args)
        
        
        for n in temp.keys():
            if task_idx>0:
                self.omega[n] = self.eta * self.omega[n]+temp[n] 
            else:
                self.omega = temp
            self.mask[n] = (self.omega[n] > 0).float()

        dummy = Policy(obs_shape,taskcla,).to(device)

        pre_name = 0
        for (name,dummy_layer),(_,layer) in zip(dummy.base.named_children(), self.actor_critic.base.named_children()):
            with torch.no_grad():
                if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                    if pre_name!=0:
                        temp = (self.omega[pre_name]>0).float()
                        if isinstance(layer, nn.Linear) and 'conv' in pre_name:
                            temp = temp.unsqueeze(0).unsqueeze(-1)
                            weight = layer.weight
                            weight = weight.view(weight.size(0), temp.size(1), -1)
                            weight = weight * temp
                            layer.weight.data = weight.view(weight.size(0), -1)
                        elif len(weight.size())>2:
                            temp = temp.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                            layer.weight *= temp
                        else:
                            temp = temp.unsqueeze(0)
                            layer.weight *= temp
                            
                    weight = layer.weight.data
                    bias = layer.bias.data
                    
                    if len(weight.size()) > 2:
                        norm = weight.norm(2,dim=(1,2,3))
                        mask = (self.omega[name]==0).float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

                    else:
                        norm = weight.norm(2,dim=(1))
                        mask = (self.omega[name]==0).float().unsqueeze(-1)

                    zero_cnt = int((mask.sum()).item())
                    indice = np.random.choice(range(zero_cnt), int(zero_cnt*(1-self.gamma)), replace=False)
                    indice = torch.tensor(indice).long()
                    idx = torch.arange(weight.shape[0])[mask.flatten(0)==1][indice]
                    mask[idx] = 0

                    layer.weight.data = (1-mask)*layer.weight.data + mask*dummy_layer.weight.data
                    mask = mask.squeeze()
                    layer.bias.data = (1-mask)*bias + mask*dummy_layer.bias.data
                    
                    pre_name = name
                if isinstance(layer, nn.ModuleList):
                    weight = layer[task_idx].weight
                    weight[:, self.omega[pre_name] == 0] = 0
                    
        self.actor_critic_old = deepcopy(self.actor_critic)
        self.actor_critic_old.train()
        self.freeze_model(self.actor_critic_old) # Freeze the weights
    





