import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .a2c_ppo_acktr.distributions import Categorical
from .a2c_ppo_acktr.utils import init

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def print_model_report(model):
    print('-'*100)
    print(model)
    print('Dimensions =',end=' ')
    count=0
    for p in model.parameters():
        print(p.size(),end=' ')
        count+=np.prod(p.size())
    print()
    print('Num parameters = %s'%(human_format(count)))
    print('-'*100)
    return count

def human_format(num):
    magnitude=0
    while abs(num)>=1000:
        magnitude+=1
        num/=1000.0
    return '%.1f%s'%(num,['','K','M','G','T','P'][magnitude])

class Policy(nn.Module):
    def __init__(self, obs_shape, taskcla, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            else:
                raise NotImplementedError

        self.taskcla = taskcla
        self.base = base(obs_shape[0], taskcla, **base_kwargs)
        print_model_report(self.base) #4.1M
        
        self.dist = torch.nn.ModuleList()

        for t,n in self.taskcla:
            self.dist.append(Categorical(self.base.output_size, n))

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks, task_num):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, task_num, deterministic=False, avg_act = False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks, task_num, avg_act)
        dist = self.dist[task_num](actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks, task_num):
        value, _, _ = self.base(inputs, rnn_hxs, masks, task_num)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, task_num):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks, task_num)
        dist = self.dist[task_num](actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs

    def evaluate_actions_(self, inputs, rnn_hxs, masks, action, task_num):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks, task_num)

        return actor_features


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, taskcla, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))
        
        self.taskcla = taskcla
        
        print ('CNN model') #4.1M

        self.conv1 = nn.Conv2d(num_inputs, 32*4, 8, stride=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32*4, 64*4, 4, stride=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64*4, 32*4, 3, stride=1)
        self.relu3 = nn.ReLU()
        self.flatten = Flatten()
        self.linear1 = nn.Linear(32 * 4 * 7 * 7, hidden_size)
        self.relu4 = nn.ReLU()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

#         self.critic_linear = init_(nn.Linear(hidden_size, 1))
        
        self.critic_linear = torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.critic_linear.append(init_(torch.nn.Linear(hidden_size,1)))

        self.train()

    def forward(self, inputs, rnn_hxs, masks, task_num, avg_act = False):

        activation1 = self.relu1(self.conv1(inputs/255.))
        activation2 = self.relu2(self.conv2(activation1))
        activation3 = self.relu3(self.conv3(activation2))
        flatten = self.flatten(activation3)
        linear = self.relu4(self.linear1(flatten))
        
        critic_output=[]
        for t,i in self.taskcla:
            critic_output.append((self.critic_linear[t](linear)))
            
        if avg_act == True:
            names = [0,  1, 2, 3]
            act = [activation1, activation2, activation3, linear]
            
            self.activations = []
            for i in act:
                self.activations.append(i.detach())

        return critic_output[task_num], linear, rnn_hxs




