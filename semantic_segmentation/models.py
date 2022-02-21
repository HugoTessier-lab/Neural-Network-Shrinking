from semantic_segmentation.networks import hrnet


def get_model(args):
    if args.model in hrnet.__all__:
        model = eval('hrnet.' + args.model + f'(pretrained={args.pretrained}, adder={args.pruning_rate != 0})')
    else:
        print('ERROR : non existing model type.')
        raise ValueError
    return model
