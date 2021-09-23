import classification.networks.resnet as resnet
import classification.networks.mobilenet as mobilenet


def get_model(args):
    if args.model not in resnet.__all__ and args.model not in mobilenet.__all__:
        print('Invalid model')
        raise ValueError
    if args.model in resnet.__all__:
        if args.dataset == 'cifar10':
            num_class = 10
        elif args.dataset == 'cifar100':
            num_class = 100
        elif args.dataset == 'imagenet':
            num_class = 1000
        else:
            print("Wrong dataset specified")
            raise ValueError
        model = resnet.resnet_model(args.model, num_class, in_planes=args.input_feature_maps)
    elif args.model == 'mobilenet':
        model = mobilenet.mobilenet_v2()
    else:
        print('ERROR : non existing classification network type.')
        raise ValueError
    return model
