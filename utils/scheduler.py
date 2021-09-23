import torch
import math


def get_scheduler(args, optimizer):
    if args.scheduler == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs // 2, 3 * (args.epochs // 3)])
    elif args.scheduler == 'poly':
        def poly_schd(e):
            return math.pow(1 - e / args.epochs, args.poly_exp)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_schd)
    else:
        print('Invalid scheduler type')
        raise ValueError
