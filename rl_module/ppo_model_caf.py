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
        print_model_report(self.base)
        
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

    def act(self, inputs, rnn_hxs, masks, task_num, deterministic=False, avg_act = False): #return_expert=True,

        value, actor_features, rnn_hxs, value_exp = self.base(inputs, rnn_hxs, masks, task_num, avg_act)

        dist = self.dist[task_num](actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks, task_num):
        value, _, _, value_exp = self.base(inputs, rnn_hxs, masks, task_num)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, task_num):
        value, actor_features, rnn_hxs, value_exp = self.base(inputs, rnn_hxs, masks, task_num)
        dist = self.dist[task_num](actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs, value_exp

    def evaluate_actions_(self, inputs, rnn_hxs, masks, action, task_num):
        value, actor_features, rnn_hxs, value_exp = self.base(inputs, rnn_hxs, masks, task_num)

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
        
        print ('CNN model')

        self.nLearner = 5
        self.nc = 32

        self.net1 = nn.Sequential(
            nn.Conv2d(num_inputs, self.nc, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(self.nc, self.nc * 2, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(self.nc * 2, self.nc, 3, stride=1),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(self.nc * 7 * 7, hidden_size) #hidden_size=512

        self.net2 = nn.Sequential(
            nn.Conv2d(num_inputs, self.nc, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(self.nc, self.nc * 2, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(self.nc * 2, self.nc, 3, stride=1),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(self.nc * 7 * 7, hidden_size) #hidden_size=512

        self.net3 = nn.Sequential(
            nn.Conv2d(num_inputs, self.nc, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(self.nc, self.nc * 2, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(self.nc * 2, self.nc, 3, stride=1),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(self.nc * 7 * 7, hidden_size) #hidden_size=512

        self.net4 = nn.Sequential(
            nn.Conv2d(num_inputs, self.nc, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(self.nc, self.nc * 2, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(self.nc * 2, self.nc, 3, stride=1),
            nn.ReLU()
        )
        self.fc4 = nn.Linear(self.nc * 7 * 7, hidden_size) #hidden_size=512

        self.net5 = nn.Sequential(
            nn.Conv2d(num_inputs, self.nc, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(self.nc, self.nc * 2, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(self.nc * 2, self.nc, 3, stride=1),
            nn.ReLU()
        )
        self.fc5 = nn.Linear(self.nc * 7 * 7, hidden_size) #hidden_size=512

        self.relu = nn.ReLU()
        self.flatten = Flatten()
        self.adaptor = nn.init.normal_(nn.Parameter(torch.ones((8, self.nLearner))))

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.critic_linear.append(init_(torch.nn.Linear(hidden_size,1)))

        self.train()

    def forward(self, inputs, rnn_hxs, masks, task_num, avg_act = False, return_expert=True):

        self.Learners = []
        self.Learners_feature = []

        h1 = self.net1(inputs/255.)
        h1 = self.flatten(h1)
        self.Learners_feature.append(h1)
        h1 = self.relu(self.fc1(h1))
        self.Learners.append(h1.unsqueeze(0))

        h2 = self.net2(inputs/255.)
        h2 = self.flatten(h2)
        self.Learners_feature.append(h2)
        h2 = self.relu(self.fc2(h2))
        self.Learners.append(h2.unsqueeze(0))

        h3 = self.net3(inputs/255.)
        h3 = self.flatten(h3)
        self.Learners_feature.append(h3)
        h3 = self.relu(self.fc3(h3))
        self.Learners.append(h3.unsqueeze(0))

        h4 = self.net4(inputs/255.)
        h4 = self.flatten(h4)
        self.Learners_feature.append(h4)
        h4 = self.relu(self.fc4(h4))
        self.Learners.append(h4.unsqueeze(0))

        h5 = self.net5(inputs/255.)
        h5 = self.flatten(h5)
        self.Learners_feature.append(h5)
        h5 = self.relu(self.fc5(h5))
        self.Learners.append(h5.unsqueeze(0))

        h = torch.cat([h_result for h_result in self.Learners], 0)
        h = torch.sum(h, dim=0).squeeze(0) #/ self.nLearner
        
        critic_output=[]
        for t,i in self.taskcla:
            critic_output.append((self.critic_linear[t](h)))

        if return_expert:
            self.Learners_y = []
            for i in range(self.nLearner):
                h_exp = self.Learners[i].squeeze(0)

                # using joint classifier
                y_exp = self.critic_linear[task_num](h_exp)
                self.Learners_y.append(y_exp)

            return critic_output[task_num], h, rnn_hxs, self.Learners_y

        else:
            return critic_output[task_num], h, rnn_hxs








