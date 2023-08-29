import sys, os, time
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim

import pickle
import torch
from arguments_rl import get_args

from collections import deque
from rl_module.a2c_ppo_acktr.envs import make_vec_envs
from rl_module.a2c_ppo_acktr.storage import RolloutStorage
from rl_module.train_ppo import train_ppo

tstart = time.time()

# Arguments
args = get_args()
conv_experiment = [
    'atari',
]

# Split
##########################################################################################################################33
if args.approach == 'fine-tuning':
    log_name = '{}_{}_{}_{}'.format(args.date, args.experiment, args.approach,args.seed)
elif args.approach == 'ewc':
    log_name = '{}_{}_{}_{}_lamb_{}'.format(args.date, args.experiment, args.approach, args.seed, args.reg_lambda)
elif args.approach == 'ewc_cpr':
    log_name = '{}_{}_{}_{}_lamb_{}_cprlamb_{}'.format(args.date, args.experiment, args.approach, args.seed, args.reg_lambda, args.cpr_lambda)
elif args.approach == 'ewc_caf':
    log_name = '{}_{}_{}_{}_lamb_{}_kldlamb_{}_aflamb_{}'.format(args.date, args.experiment, args.approach, args.seed, args.reg_lambda, args.kld_lambda, args.af_lambda)
elif args.approach == 'ewc_af2':
    log_name = '{}_{}_{}_{}_lamb_{}_aflamb_{}'.format(args.date, args.experiment, args.approach, args.seed, args.reg_lambda, args.af_lambda)
elif args.approach == 'gs':
    log_name = '{}_{}_{}_{}_lamb_{}_mu_{}'.format(args.date, args.experiment, args.approach,args.seed, args.gs_lamb, args.gs_mu)
elif args.approach == 'mas':
    log_name = '{}_{}_{}_{}_lamb_{}'.format(args.date, args.experiment, args.approach, args.seed, args.reg_lambda)
elif args.approach == 'mas_cpr':
    log_name = '{}_{}_{}_{}_lamb_{}_cprlamb_{}'.format(args.date, args.experiment, args.approach, args.seed, args.reg_lambda, args.cpr_lambda)
elif args.approach == 'mas_caf':
    log_name = '{}_{}_{}_{}_lamb_{}_kldlamb_{}_aflamb_{}'.format(args.date, args.experiment, args.approach, args.seed, args.reg_lambda, args.kld_lambda, args.af_lambda)
elif args.approach == 'mas_af2':
    log_name = '{}_{}_{}_{}_lamb_{}_aflamb_{}'.format(args.date, args.experiment, args.approach, args.seed, args.reg_lambda, args.af_lambda)
if args.single_task == True:
    log_name += '_single_task'

if args.experiment in conv_experiment:
    log_name = log_name + '_conv'

########################################################################################################################
# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
else:
    print('[CUDA unavailable]'); sys.exit()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if 'caf' in args.approach:
    from rl_module.ppo_model_caf import Policy
else:
    from rl_module.ppo_model import Policy

# Inits
print('Inits...')
torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda:0" if args.cuda else "cpu")

if args.experiment == 'atari':

    obs_shape = (4,84,84)

    taskcla = [(0, 6),(1,18),(2,18),(3,6),(4,18),(5,8),(6,9),(7,9)]

    task_sequences = [(0,'DemonAttackNoFrameskip-v4'), (1,'RobotankNoFrameskip-v4'), (2,'BoxingNoFrameskip-v4'), (3,'NameThisGameNoFrameskip-v4'),
                      (4,'StarGunnerNoFrameskip-v4'), (5,'GopherNoFrameskip-v4'), (6,'VideoPinballNoFrameskip-v4'), (7,'CrazyClimberNoFrameskip-v4')]


actor_critic = Policy(obs_shape,taskcla,).to(device)
actor_critic_emp = Policy(obs_shape,taskcla,).to(device)

# Args -- Approach
if args.approach == 'fine-tuning':
    from approaches.ppo import PPO as approach
    
    agent = approach(actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            args.optimizer,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
            use_clipped_value_loss=True)
    
elif args.approach == 'ewc':
    from approaches.ppo_ewc import PPO_EWC as approach
    
    agent = approach(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        args.optimizer,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        use_clipped_value_loss=True,
        reg_lambda=args.reg_lambda,
        online = args.ewc_online)

elif args.approach == 'ewc_cpr':
    from approaches.ppo_ewc_cpr import PPO_EWC as approach

    agent = approach(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        args.optimizer,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        use_clipped_value_loss=True,
        reg_lambda=args.reg_lambda,
        cpr_lambda=args.cpr_lambda,
        online=args.ewc_online)

elif args.approach == 'ewc_caf':
    from approaches.ppo_ewc_caf import PPO_EWC_CAF as approach

    agent = approach(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        args.optimizer,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        use_clipped_value_loss=True,
        reg_lambda=args.reg_lambda,
        online=args.ewc_online,
        af_lambda=args.af_lambda,
        kld_lambda=args.kld_lambda,
    )

elif args.approach == 'ewc_af2':
    from approaches.ppo_fg_ewc import PPO_EWC_AF2 as approach

    agent = approach(
        actor_critic,
        actor_critic_emp,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        args.optimizer,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        use_clipped_value_loss=True,
        reg_lambda=args.reg_lambda,
        af_lambda=args.af_lambda,
        online=args.ewc_online)

elif args.approach == 'mas':
    from approaches.ppo_mas import PPO_MAS as approach

    agent = approach(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        args.optimizer,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        use_clipped_value_loss=True,
        reg_lambda=args.reg_lambda,
        online=args.ewc_online)

elif args.approach == 'mas_cpr':
    from approaches.ppo_mas_cpr import PPO_MAS as approach

    agent = approach(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        args.optimizer,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        use_clipped_value_loss=True,
        reg_lambda=args.reg_lambda,
        cpr_lambda=args.cpr_lambda,
        online=args.ewc_online)

elif args.approach == 'mas_caf':
    from approaches.ppo_mas_caf import PPO_MAS_CAF as approach

    agent = approach(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        args.optimizer,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        use_clipped_value_loss=True,
        reg_lambda=args.reg_lambda,
        online=args.ewc_online,
        af_lambda=args.af_lambda,
        kld_lambda=args.kld_lambda,
    )

elif args.approach == 'mas_af2':
    from approaches.ppo_fg_mas import PPO_MAS_AF2 as approach

    agent = approach(
        actor_critic,
        actor_critic_emp,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        args.optimizer,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        use_clipped_value_loss=True,
        reg_lambda=args.reg_lambda,
        af_lambda=args.af_lambda,
        online=args.ewc_online)

elif args.approach == 'gs':
    from approaches.ppo_gs import PPO_GS  as approach
    
    agent = approach(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        args.optimizer,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        use_clipped_value_loss=True,
        mu= args.gs_mu,
        lamb= args.gs_lamb)

    
########################################################################################################################
    
tr_reward_arr = []
te_reward_arr = {}

for _type in (['mean', 'max', 'min']):
    te_reward_arr[_type] = {}
    for idx in range(len(taskcla)):
        te_reward_arr[_type]['task' + str(idx)] = []

for task_idx,env_name in task_sequences:
    print("training on task:", task_idx)
    
    envs = make_vec_envs(env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)
    obs = envs.reset()
    
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                                  obs_shape, envs.action_space,
                                  actor_critic.recurrent_hidden_state_size)
    
    if args.experiment == 'atari':
        obs_shape_real = None
        new_obs = obs

    rollouts.obs[0].copy_(new_obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    
    if task_idx > 0 and args.approach == 'gs':
        agent.freeze_init(task_idx)

    train_ppo(actor_critic, agent, rollouts, task_idx, env_name, task_sequences, envs, new_obs, obs_shape, obs_shape_real, args,
              episode_rewards, tr_reward_arr, te_reward_arr, num_updates, log_name, device)

    if args.approach == 'fine-tuning':
        if args.single_task == True:
            envs.close()
            break
        else:
            envs.close()
        
    elif args.approach == 'ewc':
        agent.update_fisher(agent, envs, rollouts, task_idx, env_name, new_obs, obs_shape_real, args)
        envs.close()

    elif args.approach == 'ewc_cpr':
        agent.update_fisher(agent, envs, rollouts, task_idx, env_name, new_obs, obs_shape_real, args)
        envs.close()

    elif args.approach == 'ewc_caf':
        agent.update_fisher(agent, envs, rollouts, task_idx, env_name, new_obs, obs_shape_real, args)
        envs.close()

    elif args.approach == 'fg_ewc':
        agent.update_fisher(agent, envs, rollouts, task_idx, env_name, new_obs, obs_shape_real, args)
        envs.close()
        
    elif args.approach == 'gs':
        
        if args.single_task == True:
            envs.close()
            break
        else:
            agent.update_omega(rollouts, task_idx, taskcla, obs_shape, device, args)
            envs.close()

    elif args.approach == 'mas':
        agent.update_omega(agent, envs, rollouts, task_idx, env_name, new_obs, obs_shape_real, args)
        envs.close()

    elif args.approach == 'mas_cpr':
        agent.update_omega(agent, envs, rollouts, task_idx, env_name, new_obs, obs_shape_real, args)
        envs.close()

    elif args.approach == 'mas_caf':
        agent.update_omega(agent, envs, rollouts, task_idx, env_name, new_obs, obs_shape_real, args)
        envs.close()

    elif args.approach == 'mas_af2':
        agent.update_omega(agent, envs, rollouts, task_idx, env_name, new_obs, obs_shape_real, args)
        envs.close()


########################################################################################################################



