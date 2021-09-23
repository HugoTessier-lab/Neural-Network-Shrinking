import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Neural Network Shrinking')

    # TASK

    parser.add_argument('--task', type=str, default="semanticsegmentation",
                        help="Type of task (either 'semanticsegmentation' or 'classification')")

    # PATHS

    parser.add_argument('--results_path', type=str, default="./results",
                        help="Where to save results  (default: './results')")

    parser.add_argument('--checkpoint_path', type=str, default="./checkpoints",
                        help="Where to save models  (default: './checkpoints')")

    parser.add_argument('--dataset_path', type=str, default="./data",
                        help="Where to get the dataset (default: './dataset')")

    # GENERAL

    parser.add_argument('--dataset', type=str, default="./cityscapes",
                        help="dataset to use  (default: './cityscapes')")

    parser.add_argument('--optimizer', type=str, default="sgd",
                        help="Type of optimizer (default: 'sgd')")

    parser.add_argument('--scheduler', type=str, default="poly",
                        help="Type of scheduler (default: 'poly')")

    parser.add_argument('--poly_exp', type=float, default=2,
                        help='Polynomial exponent of scheduler (default: 2)')

    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')

    parser.add_argument('--wd', default="5e-4", type=float,
                        help='Weight decay rate (default: 5e-4)')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='Input batch size for training (default: 1)')

    parser.add_argument('--test_batch_size', type=int, default=1,
                        help='Input batch size for testing (default: 1)')

    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train (default: 200)')

    parser.add_argument("--distributed", action="store_true", default=False,
                        help="Distributes the model across available GPUs.")

    parser.add_argument("--debug", action="store_true", default=False,
                        help="Debug mode.")

    parser.add_argument('--criterion', type=str, default="rmi",
                        help="Type of criterion (default: 'rmi')")

    parser.add_argument('--metrics', type=str, nargs='+', default=['miou'],
                        help="List of metrics (default: ['miou'])")

    parser.add_argument('--device', type=str, default="cuda",
                        help="Device to use (default: 'cuda')")

    # CLASSIFICATION

    parser.add_argument('--model', type=str, default="resnet20",
                        help="Type of classification model (default: 'resnet20')")

    parser.add_argument('--input_feature_maps', type=int, default=64,
                        help='Input feature maps of classification ResNets (either 64 or 16)')

    # SEMANTIC SEGMENTATION

    parser.add_argument('--encoder', type=str, default="hrnet48",
                        help="Type of encoder (default: 'hrnet48')")

    parser.add_argument('--decoder', type=str, default="unet",
                        help="Type of decoder (default: 'unet')")

    parser.add_argument("--unet_variant", action="store_true", default=False,
                        help="Uses the unet variant.")

    parser.add_argument("--unet_batchnorm", action="store_true", default=False,
                        help="Uses the unet batchnorms.")

    parser.add_argument("--unet_concat", action="store_true", default=False,
                        help="Unet concatenates feature maps instead of summing them.")

    parser.add_argument("--deeplab_depthwise", action="store_true", default=False,
                        help="Perform DeepLabV3 with depthwise convolutions.")

    parser.add_argument("--deeplab_decoder", action="store_true", default=False,
                        help="Extends DeepLabV3 with a decoder.")

    parser.add_argument("--deeplab_hierarchical", action="store_true", default=False,
                        help="Fuses DeepLabV3's multi-scales features in a hierarchical/residual way.")

    parser.add_argument('--output_stride', type=int, default=16,
                        help='Output stride (16 or 8) of ResNet when dilated for DeepLabV3 (default: 16)')

    parser.add_argument("--pretrained", action="store_true", default=False,
                        help="Use pretrained models.")

    return parser.parse_args()
