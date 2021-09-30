from semantic_segmentation.networks import hrnet


def get_model(args):
    if args.model in hrnet.__all__:
        model = eval('hrnet.' + args.encoder + f'(pretrained={args.pretrained})')
    else:
        print('ERROR : non existing encoder type.')
        raise ValueError
    return model
