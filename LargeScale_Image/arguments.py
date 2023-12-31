import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Continual Learning')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    # CUB: 0.005
    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='learning rate (default: 0.1. Note that lr is decayed by args.gamma parameter args.schedule ')
    parser.add_argument('--decay', type=float, default=0, help='Weight decay (L2 penalty).')
    parser.add_argument('--lamb', type=float, default=0, help='Lambda for gs, mas, rwalk, ewc, si, hat')
    parser.add_argument('--lamb_cpr', type=float, default=0, help='Lambda for cpr')
    parser.add_argument('--lamb_kld', type=float, default=0, help='Lambda for kld')
    parser.add_argument('--lamb_af', type=float, default=0, help='Lambda for af')

    parser.add_argument('--mu', type=float, default=1, help='Mu for gs')
    parser.add_argument('--eta', type=float, default=1, help='Gracefully forgetting')
    parser.add_argument('--gamma', type=float, default=0.75, help='HAT reg strength or AGS rand-init')
    parser.add_argument('--smax', type=int, default=400, help='HAT reg strength')
    parser.add_argument('--rho', type=float, default=0.1, help='Rho for GS')
    parser.add_argument('--schedule', type=int, nargs='+', default=[30],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gammas', type=float, nargs='+', default=[0.1],
                        help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seeds values to be used; seed introduces randomness by changing order of classes')
    parser.add_argument('--nepochs', type=int, default=40, help='Number of epochs for each increment')
    parser.add_argument('--tasknum', default=10, type=int, help='(default=%(default)s)')
    parser.add_argument('--date', type=str, default='', help='(default=%(default)s)')
    parser.add_argument('--output', default='', type=str, required=False, help='(default=%(default)s)')
    parser.add_argument('--s_gate', default='1', type=float, help='(default=%(default)f)')
    parser.add_argument('--dataset', default='CUB200', type=str,
                        choices=['CUB200',  'tinyImageNet' ],
                        help='(default=%(default)s)')
    
    parser.add_argument('--trainer', default='ewc', type=str,
                        choices=['ewc', 'ewc_caf',
                                 'mas', 'mas_caf',
                                 'gs', 'si', 'rwalk'], help='(default=%(default)s)')

    parser.add_argument('--adapt_af', action='store_true', help='adapt_af')
    parser.add_argument('--adapt_kld', action='store_true', help='adapt_kld')

    args = parser.parse_args()
    return args
