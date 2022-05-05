import argparse
import torch
from shrinking.onnx_conversion import to_onnx


def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Neural Network Shrinking')

    parser.add_argument('--model_path', type=str, default="",
                        help="Path to the .pt or .chk model to convert (default: '')")

    parser.add_argument('--frozen_image_shape', type=tuple_type, default=(1, 3, 1024, 2048),
                        help="Image shape for which to configure the ONNX network (default: (1, 3, 1024, 2048))")

    parser.add_argument('--device', type=str, default="cuda",
                        help="Device to use (default: 'cuda')")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    with open(args.model_path, 'rb') as f:
        model = torch.load(f).to(args.device)
    model.freeze(args.frozen_image_shape)
    to_onnx(model, args.model_path.replace('.pt', '.onnx'), args.frozen_image_shape, args.device)
