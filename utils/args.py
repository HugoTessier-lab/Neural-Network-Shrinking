import argparse


def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Neural Network Shrinking')

    # PATHS

    parser.add_argument('--results_path', type=str, default="./results",
                        help="Where to save results  (default: './results')")

    parser.add_argument('--checkpoint_path', type=str, default="./checkpoints",
                        help="Where to save models  (default: './checkpoints')")

    parser.add_argument('--dataset_path', type=str, default="./data",
                        help="Where to get the dataset (default: './dataset')")

    # GENERAL

    parser.add_argument('--model', type=str, default="resnet20",
                        help="Type of model (default: 'resnet20')")

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

    parser.add_argument('--frozen_image_shape', type=tuple_type, default=(1, 3, 512, 1024),
                        help="Image shape for which to freeze the network at the end (default: (1, 3, 512, 1024))")

    # CLASSIFICATION

    parser.add_argument('--input_feature_maps', type=int, default=64,
                        help='Input feature maps of classification ResNets (either 64 or 16)')

    # SEMANTIC SEGMENTATION

    parser.add_argument("--pretrained", action="store_true", default=False,
                        help="Use pretrained models.")

    # PRUNING METHOD

    parser.add_argument('--pruning_rate', type=float, default=0.,
                        help='Pruning rate (default: 0.)')

    parser.add_argument('--pruner_image_shape', type=tuple_type, default=(1, 3, 64, 64),
                        help="Image shape for the pruner's inference (default: (1, 3, 64, 64))")

    parser.add_argument("--lrr", action="store_true", default=False,
                        help="LR-Rewinding.")

    parser.add_argument('--pruning_criterion', type=str, default='global',
                        help='Pruning criterion (default: "global")')

    parser.add_argument('--pruning_method', type=str, default='swd',
                        help='Pruning method (default: "swd")')

    parser.add_argument('--pruning_iterations', type=int, default=1,
                        help='Number of iterations in which divide pruning; works only with liu2017 (default: 1)')

    parser.add_argument('--fine_tuning_epochs', type=int, default=20,
                        help='Number of fine-tuning epochs for each pruning iteration (default: 20)')

    parser.add_argument('--additional_final_epochs', type=int, default=0,
                        help="How many more fine-tuning epochs at the very end (default: 0)")

    # SWD

    parser.add_argument('--a_min', type=float, default=1e0,
                        help='Starting value for the a of SWD (default: 1e0)')

    parser.add_argument('--a_max', type=float, default=1e0,
                        help='Final value for the a of SWD (default: 1e0)')

    # LIU2017

    parser.add_argument('--liu2017_penalty', type=float, default=1e-6,
                        help='Value of the smooth-L1 penalty for Liu2017 (default: 1e-6)')

    return parser.parse_args()
