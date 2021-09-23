from semantic_segmentation.encoders import mobilenetv2, hrnet, resnet, vgg, shufflenet
from semantic_segmentation.decoders import fcn, deeplabv3, unet


def get_model(args):
    if args.encoder in vgg.__all__:
        encoder = eval('vgg.' + args.encoder + f'(pretrained={args.pretrained})')
    elif args.encoder in resnet.__all__:
        if args.output_stride is None:
            encoder = eval('resnet.' + args.encoder + f'(pretrained={args.pretrained})')
        else:
            encoder = eval('resnet.' + args.encoder + f'(pretrained={args.pretrained}, '
                                                      f'output_stride={args.output_stride})')
    elif args.encoder in mobilenetv2.__all__:
        encoder = eval('mobilenetv2.' + args.encoder + f'(pretrained={args.pretrained})')
    elif args.encoder in shufflenet.__all__:
        encoder = eval('shufflenet.' + args.encoder + f'(pretrained={args.pretrained})')
    elif args.encoder in hrnet.__all__:
        encoder = eval('hrnet.' + args.encoder + f'(pretrained={args.pretrained})')
    else:
        print('ERROR : non existing encoder type.')
        raise ValueError

    if not isinstance(encoder, hrnet.HighResolutionNet):
        if args.decoder == 'fcn8':
            model = fcn.FCN8(encoder)
        elif args.decoder == 'unet':
            model = unet.Unet(encoder, variant=args.unet_variant, batchnorm=args.unet_batchnorm,
                              concat=args.unet_concat)
        elif args.decoder == 'deeplabv3':
            model = deeplabv3.DeepLabV3(encoder, depthwise=args.deeplab_depthwise, decoder=args.deeplab_decoder,
                                        hierarchical=args.deeplab_hierarchical)
        else:
            print('ERROR : non existing decoder type.')
            raise ValueError
    else:
        model = encoder
    return model
