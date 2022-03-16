import argparse
import pickle
from utils.models import get_model
import torch
from shrinking.onnx_conversion import to_onnx
from shrinking.pruner import Pruner
from pruning.target_calculation import find_mask
from pruning.pruning_criterion import get_pruning_criterion


def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Neural Network Shrinking')

    parser.add_argument('--model_path', type=str, default="",
                        help="Path to the .pt or .chk model to convert (default: '')")

    parser.add_argument('--model', type=str, default="resnet20",
                        help="Type of model (default: 'resnet20')")

    parser.add_argument('--frozen_image_shape', type=tuple_type, default=(1, 3, 512, 1024),
                        help="Image shape for which to configure the ONNX network (default: (1, 3, 512, 1024))")

    parser.add_argument('--dataset', type=str, default="./cityscapes",
                        help="dataset to use  (default: './cityscapes')")

    parser.add_argument("--pretrained", action="store_true", default=False,
                        help="Use pretrained models.")

    parser.add_argument('--pruning_rate', type=float, default=0.,
                        help='Pruning rate (default: 0.)')

    parser.add_argument('--input_feature_maps', type=int, default=64,
                        help='Input feature maps of classification ResNets (either 64 or 16)')

    parser.add_argument('--device', type=str, default="cuda",
                        help="Device to use (default: 'cuda')")

    parser.add_argument('--pruning_criterion', type=str, default='global',
                        help='Pruning criterion (default: "global")')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    with open(args.model_path, 'rb') as f:
        if args.model_path.endswith('.chk'):
            data = pickle.load(f)
            state_dict = data['model']
            model = get_model(args).to(args.device)
            model.load_state_dict(state_dict)
            pruner = Pruner(model, args.frozen_image_shape, args.device)
            mask = find_mask(model, pruner, args.pruning_rate, get_pruning_criterion(args.pruning_criterion))
            pruner.apply_mask(mask)
            pruner.shrink_model(model)
            model.freeze(args.frozen_image_shape)
            name = args.model_path[:-3] + 'onnx'
        elif args.model_path.endswith('.pt'):
            model = torch.load(f)
            name = args.model_path[:-2] + 'onnx'
        else:
            print('Invalid model extension')
            raise ValueError

    to_onnx(model, name, args.frozen_image_shape, args.device)
