from pruning.swd import SWD


def get_pruning_method(args, model, dataset):
    if args.pruning_method == 'none':
        return None
    elif args.pruning_method == 'swd':
        return SWD(model, dataset, args.epochs, args.a_min, args.a_max, args.pruning_rate,
                   args.pruner_image_shape, args.wd)
    else:
        print('ERROR : non existing pruning method type.')
        raise ValueError
