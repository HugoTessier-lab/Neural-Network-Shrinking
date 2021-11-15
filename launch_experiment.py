from utils import args, checkpoint, optimizer, scheduler, train


def main():
    arguments = args.parse_arguments()

    if arguments.task == 'semanticsegmentation':
        from semantic_segmentation import criterion, datasets, metrics, models
    elif arguments.task == 'classification':
        from classification import criterion, datasets, metrics, models
    else:
        print('Wrong task type.')
        raise ValueError

    model = models.get_model(arguments)
    optim = optimizer.get_optimizer(arguments, model)
    sched = scheduler.get_scheduler(arguments, optim)
    check = checkpoint.Checkpoint(model, optim, sched, arguments.device, arguments.distributed,
                                  arguments.checkpoint_path)

    dataset = datasets.get_dataset(arguments)
    met = [metrics.get_metric(n) for n in arguments.metrics]
    crit = criterion.get_criterion(arguments.criterion)

    train.train_model(checkpoint=check, dataset=dataset,
                      epochs=arguments.epochs, criterion=crit, metrics=met,
                      output_path=arguments.results_path, debug=arguments.debug)


if __name__ == '__main__':
    main()
