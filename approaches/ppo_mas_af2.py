import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from copy import deepcopy
import itertools
import numpy as np


class PPO_MAS_AF2():
    def __init__(self,
                 actor_critic,
                 actor_critic_emp,
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
                 mas_epoch = 1,
                 reg_lambda= 5000,
                 af_lambda = 1,
                 online = False):

        self.actor_critic = actor_critic
        self.actor_critic_emp = deepcopy(actor_critic) #actor_critic_emp

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        
        self.mas_epoch = 1
        self.reg_lambda = reg_lambda

        self.af_lambda = af_lambda
        
        print ('reg_lambda : ', self.reg_lambda)

        if optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(itertools.chain(self.actor_critic.parameters(), self.vae.parameters()), lr=lr, momentum=0.9)
            self.optimizer_emp = torch.optim.SGD(self.actor_critic_emp.parameters(), lr=lr, momentum=0.9)
        elif optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(itertools.chain(self.actor_critic.parameters(), self.vae.parameters()), lr=lr, eps=eps)
            self.optimizer_emp = torch.optim.Adam(self.actor_critic_emp.parameters(), lr=lr, eps=eps)
        
        self.mas_task_count = 0
        self.online = online

    def update(self, rollouts, task_num):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
                data_generator_emp = rollouts.recurrent_generator(advantages, self.num_mini_batch)
                data_generator_fim = rollouts.recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)
                data_generator_emp = rollouts.feed_forward_generator(advantages, self.num_mini_batch)
                data_generator_fim = rollouts.feed_forward_generator(advantages, self.num_mini_batch)

            #checked, here is good
            #train emp synapse
            for sample in data_generator_emp:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic_emp.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch, task_num)

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                                         (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer_emp.zero_grad()
                #reg_loss = self.reg_lambda * self.mas_loss()

                loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef
                loss.backward()

                nn.utils.clip_grad_norm_(self.actor_critic_emp.parameters(),
                                         self.max_grad_norm)
                self.optimizer_emp.step()

            omega_emp = {}
            for n, p in self.actor_critic_emp.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    omega_emp[n] = p.detach().clone().zero_()

            for sample in data_generator_fim:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic_emp.evaluate_actions(
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
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer_emp.zero_grad()
                #reg_loss = self.reg_lambda * self.mas_loss()

                loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef
                loss.backward()

                for n, p in self.actor_critic_emp.named_parameters():
                    if p.requires_grad:
                        n = n.replace('.', '__')
                        if p.grad is not None:
                            omega_emp[n] += p.grad.detach() ** 2

            for n, p in self.actor_critic_emp.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    omega_emp[n] = omega_emp[n] / 100

            self.store_omega_n_params_emp(omega_emp)

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

                reg_loss = self.reg_lambda * self.mas_loss()
                fg_loss = self.af_lambda * self.fg_loss()

                self.optimizer.zero_grad()

                loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef + reg_loss + fg_loss

                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

            self.lam = self.lam * self.weight_decay
            self.lam2 = self.lam2 * self.weight_decay

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    def estimate_omega(self, rollouts, omega_dict, task_num):

        est_omega_info = omega_dict

        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.mas_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ = sample

                # simple
                # Reshape to do in a single forward pass for all steps
                actor_features = self.actor_critic.evaluate_actions_(obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch, task_num)
                self.optimizer.zero_grad()
                loss = torch.sum(actor_features.norm(2, dim=-1))
                loss.backward()

                for n, p in self.actor_critic.named_parameters():
                    if p.requires_grad:
                        n = n.replace('.', '__')
                        if p.grad is not None:
                            est_omega_info[n] += p.grad.detach().abs()

        return est_omega_info

    def store_omega_n_params(self, omega):
        # Store new values in the network
        for n, p in self.actor_critic.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.actor_critic.register_buffer('{}_mas_prev_task{}'.format(n, "" if self.online else self.mas_task_count + 1),
                                     p.detach().clone())
                if self.online and self.mas_task_count == 1:
                    existing_values = getattr(self.actor_critic, '{}_mas_estimated_omega'.format(n))
                    omega[n] += self.gamma * existing_values
                self.actor_critic.register_buffer('{}_mas_estimated_omega{}'.format(n, "" if self.online else self.mas_task_count + 1), omega[n])

        self.mas_task_count = 1 if self.online else self.mas_task_count + 1
        
    def update_omega(self, agent, envs, rollouts, task_idx, env_name, new_obs, obs_shape_real, args):
        
        omega_info={}
        
        for n, p in self.actor_critic.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                omega_info[n] = p.detach().clone().zero_()
                
                
        for batch in tqdm(range(args.mas_epochs)):
            for step in range(args.num_mas_steps):
                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = self.actor_critic.act(
                        rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step], task_idx)
                    
                if env_name == 'MinitaurBulletEnv-v0':
                    action = torch.clamp(action, -1, 1) # for MinitaurBulletEnv

                # Obser reward and next obs
                obs, reward, done, infos = envs.step(action)
                
                if args.experiment == 'roboschool':
                #### reshape for traiing ###############
                    new_obs[:, :obs_shape_real[0]] = obs
                ########################################
                else:
                    new_obs = obs


                # If done then clean the history of observations.
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                     for info in infos])
                
                rollouts.insert(new_obs, recurrent_hidden_states, action,
                                action_log_prob, value, reward, masks, bad_masks)
                
            with torch.no_grad():
                next_value = self.actor_critic.get_value(
                    rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1], task_idx).detach()

            rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                     args.gae_lambda, args.use_proper_time_limits)

            omega_info = agent.estimate_omega(rollouts, omega_info, task_idx)

#             rollouts.after_update()

        for n, p in self.actor_critic.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                omega_info[n] = omega_info[n] / 100

        agent.store_omega_n_params(omega_info)
        print('omega computed successfully')

    def store_omega_n_params_emp(self, omega):
        # Store new values in the network
        for n, p in self.actor_critic_emp.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                # -mode (=MAP parameter estimate)
                self.actor_critic_emp.register_buffer(
                    '{}_mas_prev_task{}_emp'.format(n, "" if self.online else self.mas_task_count + 1),
                    p.detach().clone())
                # -precision (approximated by diagonal omega Information matrix)
                if self.online and self.mas_task_count == 1:
                    existing_values = getattr(self.actor_critic_emp, '{}_mas_estimated_omega_emp'.format(n))
                    omega[n] += self.gamma * existing_values
                self.actor_critic_emp.register_buffer(
                    '{}_mas_estimated_omega{}_emp'.format(n, "" if self.online else self.mas_task_count + 1),
                    omega[n])

    def mas_loss(self):
        '''Calculate MAS-loss.'''
        if self.mas_task_count > 0:
            losses = []
            for task in range(1, self.mas_task_count + 1):
                for n, p in self.actor_critic.named_parameters():
                    if p.requires_grad:
                        n = n.replace('.', '__')
                        mean = getattr(self.actor_critic,
                                       '{}_mas_prev_task{}'.format(n, "" if self.online else task))
                        omega = getattr(self.actor_critic,
                                         '{}_mas_estimated_omega{}'.format(n, "" if self.online else task))
                        omega = self.gamma * omega if self.online else omega
                        # Calculate MAS-loss
                        losses.append((omega * (p - mean) ** 2).sum())
            return (1. / 2) * sum(losses)
        else:
            return 0.

    def fg_loss(self):
        '''Calculate MAS-loss.'''
        if self.mas_task_count > 0:
            losses = []
            for n, p in self.actor_critic.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    mean = getattr(self.actor_critic_emp, '{}_mas_prev_task{}_emp'.format(n, "" if self.online else self.mas_task_count + 1))
                    omega = getattr(self.actor_critic_emp, '{}_mas_estimated_omega{}_emp'.format(n, "" if self.online else self.mas_task_count + 1))
                    omega = self.gamma * omega if self.online else omega
                    # Calculate MAS-loss
                    losses.append((omega * (p - mean) ** 2).sum())
            return (1. / 2) * sum(losses)
        else:
            return 0.







