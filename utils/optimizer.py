import torch


def get_optimizer(args, model):
    if args.optimizer == 'adam':
        return torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == 'sgd':
        return torch.optim.SGD(params=model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9, nesterov=False)
    else:
        print('Invalid optimizer type')
        raise ValueError
