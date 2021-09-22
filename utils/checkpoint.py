import pickle
from utils.optimizer import get_optimizer
from utils.scheduler import get_scheduler
from models import mobilenetv2, resnet, shufflenet, vgg, hrnet
from meta import fcn, unet, deeplabv3
import torch
import os


def get_encoder(name, pretrained, output_stride):
    if name in vgg.__all__:
        return eval('vgg.' + name + f'(pretrained={pretrained})')
    elif name in resnet.__all__:
        if output_stride is None:
            return eval('resnet.' + name + f'(pretrained={pretrained})')
        else:
            return eval('resnet.' + name + f'(pretrained={pretrained}, output_stride={output_stride})')
    elif name in mobilenetv2.__all__:
        return eval('mobilenetv2.' + name + f'(pretrained={pretrained})')
    elif name in shufflenet.__all__:
        return eval('shufflenet.' + name + f'(pretrained={pretrained})')
    elif name in hrnet.__all__:
        return eval('hrnet.' + name + f'(pretrained={pretrained})')
    else:
        print('ERROR : non existing encoder type.')
        raise ValueError


def get_decoder(encoder, name, variant, batchnorm, concat, depthwise, decoder, hierarchical):
    if name == 'fcn8':
        return fcn.FCN8(encoder)
    elif name == 'unet':
        return unet.Unet(encoder, variant=variant, batchnorm=batchnorm, concat=concat)
    elif name == 'deeplabv3':
        return deeplabv3.DeepLabV3(encoder, depthwise=depthwise, decoder=decoder, hierarchical=hierarchical)


class Checkpoint:
    def __init__(self, args):
        encoder = get_encoder(args.encoder, args.pretrained,
                              args.output_stride if args.decoder == 'deeplabv3' else None)
        if not isinstance(encoder, hrnet.HighResolutionNet):
            model = get_decoder(encoder, args.decoder, variant=args.unet_variant,
                                batchnorm=args.unet_batchnorm, concat=args.unet_concat,
                                depthwise=args.deeplab_depthwise, decoder=args.deeplab_decoder,
                                hierarchical=args.deeplab_hierarchical)
        else:
            model = encoder
        self.model = model.cuda()
        if args.distributed:
            self.model = torch.nn.DataParallel(model)
        self.optimizer = get_optimizer(args.optimizer, model, args.lr, args.wd)
        self.scheduler = get_scheduler(self.optimizer, args.scheduler, args.epochs, poly_exp=args.poly_exp)

    def save_model(self, path, epoch):
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model.state_dict(),
                         'epoch': epoch,
                         'optimizer': self.optimizer.state_dict()}, f)

    def store_model(self, path, epoch):
        if not os.path.isdir(path):
            try:
                os.mkdir(path)
            except OSError:
                print(f'Failed to create the folder {path}')
            else:
                print(f'Created folder {path}')
        path = os.path.join(path, 'model.chk')
        if not os.path.isfile(path):
            self.save_model(path, epoch)
            return epoch
        else:
            with open(path, 'rb') as f:
                checkpoint = pickle.load(f)
            loaded_epoch = checkpoint['epoch']
            if epoch >= loaded_epoch:
                self.save_model(path, epoch)
                return epoch
            elif epoch < loaded_epoch:
                self.model.load_state_dict(checkpoint['model'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.last_epoch = loaded_epoch
                return loaded_epoch
            else:
                raise ValueError
