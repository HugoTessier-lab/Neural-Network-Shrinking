from utils.networks import hrnet, resnet


def get_model(args):
    if args.model in hrnet.__all__:
        model = eval('hrnet.' + args.model + f'(pretrained={args.pretrained}, adder={args.pruning_rate != 0})')
    elif args.model in resnet.__all__:
        if args.dataset == 'cifar10':
            num_class = 10
        elif args.dataset == 'cifar100':
            num_class = 100
        elif args.dataset == 'imagenet':
            num_class = 1000
        else:
            print("Wrong dataset specified")
            raise ValueError
        model = resnet.resnet_model(args.model, num_class, in_planes=args.input_feature_maps,
                                    adder=args.pruning_rate != 0)
    else:
        print('ERROR : non existing model type.')
        raise ValueError
    return model
