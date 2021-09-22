import torch
import math


def get_scheduler(optimizer, scheduler, epochs, poly_exp=2):
    if scheduler == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs // 2, 3 * (epochs // 3)])
    elif scheduler == 'poly':
        def poly_schd(e):
            return math.pow(1 - e / epochs, poly_exp)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_schd)
    else:
        print('Invalid scheduler type')
        raise ValueError
