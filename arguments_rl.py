import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Continual')
    # Arguments
    parser.add_argument('--seed', type=int, default=0, help='(default=%(default)d)')
    parser.add_argument('--experiment', default='atari', type=str, required=False,
                        choices=['atari'],
                        help='(default=%(default)s)')
    parser.add_argument('--approach', default='gs', type=str, required=False,
                        choices=['fine-tuning', 'ewc', 'ewc_cpr', 'ewc_caf', 'ewc_af2',
                                 'mas', 'mas_cpr', 'mas_caf', 'mas_af2', 'gs'], help='(default=%(default)s)')
    parser.add_argument('--optimizer', default='Adam', type=str, required=False,
                        choices=['Adam'],
                        help='(default=%(default)s)')
    parser.add_argument('--num-processes', default=128, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--lr', default=0.0003, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--date', type=str, default='', help='(default=%(default)s)')
    
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=10,
        help='number of ppo epochs (default: 10)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=64,
        help='number of batches for ppo (default: 64)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=True,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.00,
        help='entropy term coefficient (default: 0.0)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--log-dir',
        default='./weights/logs/',
        help='directory to save agent logs (default: ./weights/log/)')
    parser.add_argument(
        '--cuda',
        action='store_true',
        default=True,
        help='enables CUDA training')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=64,
        help='number of forward steps (default: 64)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=5000000,
        help='number of environment steps to train (default: 5000000)')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=True,
        help='use a linear schedule on the learning rate')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=True,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=10,
        help='save interval, one save per n updates (default: 10)')
    parser.add_argument(
        '--save-dir',
        default='./weights/',
        help='directory to save agent weights (default: ./trained_models/)')
    parser.add_argument(
        '--algo', default='ppo', help='algorithm to use: ppo')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=1000,
        help='log interval, one log per n updates (default: 1000)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=20,
        help='eval interval, one eval per n updates (default: 20)')
    parser.add_argument(
        '--reg_lambda',
        type=float,
        default=5000,
        help='lambda for stabiliity regularization')
    parser.add_argument(
        '--ewc-online',
        default=False,
        help='True == online EWC')
    parser.add_argument(
        '--ewc-epochs',
        type=int,
        default=100,
        help='epochs for EWC')
    parser.add_argument(
        '--mas-epochs',
        type=int,
        default=100,
        help='epochs for MAS')
    parser.add_argument(
        '--kld_lambda',
        type=float,
        default=0,
        help='lambda for kld')
    parser.add_argument(
        '--af_lambda',
        type=float,
        default=0,
        help='lambda for af')
    parser.add_argument(
        '--cpr_lambda',
        type=float,
        default=0,
        help='lambda for cpr')
    parser.add_argument(
        '--num-ewc-steps',
        type=int,
        default=20,
        help='epochs for EWC')
    parser.add_argument(
        '--num-mas-steps',
        type=int,
        default=20,
        help='epochs for MAS')
    parser.add_argument(
        '--gs-mu',
        type=float,
        default=0.5,
        help='mu for gs')
    parser.add_argument(
        '--gs-eta',
        type=float,
        default=1.0,
        help='eta for gs')
    parser.add_argument(
        '--gs-rho',
        type=float,
        default=1.0,
        help='rho for gs')
    parser.add_argument(
        '--gs-lamb',
        type=int,
        default=100,
        help='lamb for gs')
    parser.add_argument(
        '--gs-epochs',
        type=int,
        default=100,
        help='epochs for GS')
    parser.add_argument(
        '--num-gs-steps',
        type=int,
        default=20,
        help='epochs for GS')
    parser.add_argument(
        '--single-task',
        action='store_true',
        default=False,
        help='only train single task')
    args=parser.parse_args()
    return args

