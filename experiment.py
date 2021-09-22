from utils import args, datasets, train, swd
from utils.checkpoint import Checkpoint


def main():
    arguments = args.parse_arguments()
    dataset = datasets.load_cityscapes(path=arguments.dataset_path,
                                       batch_size=arguments.batch_size,
                                       test_batch_size=arguments.test_batch_size)
    checkpoint = Checkpoint(arguments)
    if arguments.swd:
        pruner = swd.SWD(len(dataset['train']), arguments.epochs,
                         arguments.a_min, arguments.a_max, arguments.pruning_rate)
    else:
        pruner = None
    train.train_model(checkpoint=checkpoint, dataset=dataset,
                      epochs=arguments.epochs, output_path=arguments.output_path,
                      debug=arguments.debug, criterion=arguments.criterion,
                      swd=pruner, print_images=arguments.print_images)


if __name__ == '__main__':
    main()
