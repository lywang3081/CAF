import sys, os, time
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim

import pickle
import utils
import torch
from arguments import get_args

# Arguments

args = get_args()

##########################################################################################################################33

if args.approach == 'gs':
    log_name = '{}_{}_{}_{}_lamb_{}_mu_{}_rho_{}_eta_{}_lr_{}_batch_{}_epoch_{}'.format(args.date, args.experiment,
                                                                                          args.approach, args.seed,
                                                                                          args.lamb, args.mu, args.rho, args.eta,
                                                                                          args.lr, args.batch_size, args.nepochs)
elif args.approach == 'hat':
    log_name = '{}_{}_{}_{}_gamma_{}_smax_{}_lr_{}_batch_{}_epoch_{}'.format(args.date, args.experiment, args.approach,
                                                                             args.seed, args.gamma, args.smax, args.lr,
                                                                             args.batch_size, args.nepochs)
else:
    log_name = '{}_{}_{}_{}_lamb_{}_lr_{}_batch_{}_epoch_{}'.format(args.date, args.experiment, args.approach, args.seed,
                                                                    args.lamb, args.lr, args.batch_size, args.nepochs)

if args.output == '':
    args.output = './result_data/' + log_name + '.txt'
tr_output = './result_data/' + log_name + '_train' '.txt'
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

# Args -- Experiment
if args.experiment == 's_cifar100':
    from dataloaders import s_cifar100 as dataloader
elif args.experiment == 'r_cifar100':
    from dataloaders import r_cifar100 as dataloader
elif args.experiment == 'r_cifar10_100':
    from dataloaders import r_cifar10_100 as dataloader
elif args.experiment == 'omniglot':
    from dataloaders import omniglot as dataloader

# Args -- Approach
if args.approach == 'gs':
    from approaches import gs as approach
elif args.approach == 'ewc':
    from approaches import ewc as approach
elif args.approach == 'ewc_caf':
    from approaches import ewc_caf as approach
elif args.approach == 'ewc_af2':
    from approaches import ewc_af2 as approach
elif args.approach == 'mas':
    from approaches import mas as approach
elif args.approach == 'mas_caf':
    from approaches import mas_caf as approach
elif args.approach == 'mas_af2':
    from approaches import mas_af2 as approach
elif args.approach == 'random_init':
    from approaches import random_init as approach
elif args.approach == 'si':
    from approaches import si as approach
elif args.approach == 'rwalk':
    from approaches import rwalk as approach
elif args.approach == 'hat':
    from approaches import hat as approach
elif args.approach == 'er':
    from approaches import er as approach
elif args.approach == 'ft':
    from approaches import ft as approach


if  args.experiment == 'omniglot':
    if args.approach == 'hat':
        from networks import conv_net_omniglot_hat as network
    elif 'caf' in args.approach:
        from networks import conv_net_omniglot_caf as network
    else:
        from networks import conv_net_omniglot as network
else:
    if args.approach == 'hat':
        from networks import conv_net_hat as network
    elif 'caf' in args.approach:
        if args.mcl == 'mcl-l': #low diversity background, remove dropoout
            from networks import conv_net_nodrop_caf as network
        else: #keep dropout
            from networks import conv_net_caf as network
    else:
        from networks import conv_net as network


########################################################################################################################


# Load
print('Load data...')
data, taskcla, inputsize = dataloader.get(seed=args.seed, tasknum=args.tasknum) # num_task is provided by dataloader
print('\nInput size =', inputsize, '\nTask info =', taskcla)

# Inits
print('Inits...')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

if not os.path.isdir('result_data'):
    print('Make directory for saving results')
    os.makedirs('result_data')
    
if not os.path.isdir('trained_model'):
    print('Make directory for saving trained models')
    os.makedirs('trained_model')

if 'caf' in args.approach:
    if args.original_width and 'cifar' in args.experiment:
        net = network.Net(inputsize, taskcla, args.use_sigmoid, nc=32).cuda()
    else:
        net = network.Net(inputsize, taskcla, args.use_sigmoid).cuda()
elif 'af2' in args.approach:
    net = network.Net(inputsize, taskcla).cuda()
    net_emp = network.Net(inputsize, taskcla).cuda()
else:
    net = network.Net(inputsize, taskcla).cuda()

if 'caf' in args.approach:
    appr = approach.Appr(net, sbatch=args.batch_size, lr=args.lr, nepochs=args.nepochs, args=args, log_name=log_name, use_sigmoid = args.use_sigmoid)
elif 'af2' in args.approach:
    appr = approach.Appr(net, sbatch=args.batch_size, lr=args.lr, nepochs=args.nepochs, args=args, log_name=log_name, empty_net = net_emp)
else:
    appr = approach.Appr(net, sbatch=args.batch_size, lr=args.lr, nepochs=args.nepochs, args=args, log_name=log_name)

utils.print_model_report(net)
print(appr.criterion)
utils.print_optimizer_config(appr.optimizer)
print('-' * 100)
relevance_set = {}
# Loop tasks
acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)


for t, ncla in taskcla:
    if t==1 and 'find_mu' in args.date:
        break
    
    print('*' * 100)
    print('Task {:2d} ({:s})'.format(t, data[t]['name']))
    print('*' * 100)

    # Get data
    xtrain = data[t]['train']['x'].cuda()
    xvalid = data[t]['valid']['x'].cuda()

    ytrain = data[t]['train']['y'].cuda()
    yvalid = data[t]['valid']['y'].cuda()
    task = t

    # Train
    appr.train(task, xtrain, ytrain, xvalid, yvalid, data, inputsize, taskcla)
    print('-' * 100)

    # Test
    for u in range(t + 1):
        xtest = data[u]['test']['x'].cuda()
        ytest = data[u]['test']['y'].cuda()

        test_loss, test_acc = appr.eval(u, xtest, ytest)
        print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u, data[u]['name'], test_loss, 100 * test_acc))

        acc[t, u] = test_acc
        lss[t, u] = test_loss


    # Save
    print('Average accuracy={:5.1f}%'.format(100 * np.mean(acc[t,:t+1])))
    print('Save at ' + args.output)
    np.savetxt(args.output, acc, '%.4f')


# Done
print('*' * 100)
print('Accuracies =')
for i in range(acc.shape[0]):
    print('\t', end='')
    for j in range(acc.shape[1]):
        print('{:5.1f}% '.format(100 * acc[i, j]), end='')
    print()
print('*' * 100)
print('Done!')


if hasattr(appr, 'logs'):
    if appr.logs is not None:
        # save task names
        from copy import deepcopy

        appr.logs['task_name'] = {}
        appr.logs['test_acc'] = {}
        appr.logs['test_loss'] = {}
        for t, ncla in taskcla:
            appr.logs['task_name'][t] = deepcopy(data[t]['name'])
            appr.logs['test_acc'][t] = deepcopy(acc[t, :])
            appr.logs['test_loss'][t] = deepcopy(lss[t, :])
        # pickle
        import gzip
        import pickle

        with gzip.open(os.path.join(appr.logpath), 'wb') as output:
            pickle.dump(appr.logs, output, pickle.HIGHEST_PROTOCOL)

########################################################################################################################

