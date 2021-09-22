import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='SWD')

    parser.add_argument('--output_path', type=str, default="./outputs",
                        help="Where to save models  (default: './checkpoint')")

    parser.add_argument("--print_images", action="store_true", default=False,
                        help="Print one image at each epoch.")

    parser.add_argument('--dataset_path', type=str, default="./data",
                        help="Where to get the dataset (default: './dataset')")

    parser.add_argument('--encoder', type=str, default="hrnet48",
                        help="Type of encoder (default: 'hrnet48')")

    parser.add_argument('--decoder', type=str, default="unet",
                        help="Type of decoder (default: 'unet')")

    parser.add_argument('--criterion', type=str, default="rmi",
                        help="Type of criterion (default: 'rmi')")

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

    parser.add_argument("--debug", action="store_true", default=False,
                        help="Debug mode.")

    parser.add_argument("--pretrained", action="store_true", default=False,
                        help="Use pretrained models.")

    parser.add_argument("--distributed", action="store_true", default=False,
                        help="Distributes the model across available GPUs.")

    parser.add_argument("--swd", action="store_true", default=False,
                        help="Prunes model using SWD.")

    parser.add_argument('--a_min', default=1, type=float,
                        help='SWD lower bound.')

    parser.add_argument('--a_max', default=1e5, type=float,
                        help='SWD higher bound.')

    parser.add_argument('--pruning_rate', default=0.5, type=float,
                        help='Pruning rate.')

    return parser.parse_args()
