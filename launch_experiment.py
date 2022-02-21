from utils import args, checkpoint, optimizer, scheduler, train
from pruning.pruning_criterion import get_pruning_criterion
from pruning.swd import SWD
import torch
import os


def main():
    arguments = args.parse_arguments()

    if arguments.task == 'semanticsegmentation':
        from semantic_segmentation import criterion, datasets, metrics, models
    elif arguments.task == 'classification':
        from classification import criterion, datasets, metrics, models
    else:
        print('Wrong task type.')
        raise ValueError

    model = models.get_model(arguments).to(arguments.device)
    optim = optimizer.get_optimizer(arguments, model)
    sched = scheduler.get_scheduler(arguments, optim)
    check = checkpoint.Checkpoint(model, optim, sched, arguments.device, arguments.distributed,
                                  arguments.checkpoint_path,
                                  f'base_swd_{arguments.pruning_rate}_{arguments.a_min}_{arguments.a_max}')

    dataset = datasets.get_dataset(arguments)
    met = [metrics.get_metric(n) for n in arguments.metrics]
    crit = criterion.get_criterion(arguments.criterion)

    swd = None
    if arguments.pruning_rate != 0:
        swd = SWD(model, dataset, arguments.epochs, arguments.a_min, arguments.a_max, arguments.pruning_rate,
                  arguments.pruner_image_shape, arguments.wd, get_pruning_criterion(arguments.pruning_criterion))

    count = len(torch.cat([i.flatten() for i in check.model.parameters()]))
    print('Original count of parameters: ', count)

    train.train_model(name='Training', checkpoint=check, dataset=dataset,
                      epochs=arguments.epochs, criterion=crit, metrics=met,
                      output_path=arguments.results_path, debug=arguments.debug,
                      device=arguments.device,
                      swd=swd)

    if swd:
        swd.prune()

    new_count = len(torch.cat([i.flatten() for i in check.model.parameters()]))
    print('\nCount of parameters after pruning: ', new_count)
    print('Ratio: ', round((new_count / count) * 100, 2))

    if arguments.lrr:
        check.optimizer = optimizer.get_optimizer(arguments, check.model)
        check.scheduler = scheduler.get_scheduler(arguments, check.optimizer)
        check.name = f'LRR_swd_{arguments.pruning_rate}_{arguments.a_min}_{arguments.a_max}'
        train.train_model(name=f'LRR_{arguments.pruning_rate}', checkpoint=check, dataset=dataset,
                          epochs=arguments.epochs, criterion=crit, metrics=met,
                          output_path=arguments.results_path, debug=arguments.debug,
                          device=arguments.device,
                          swd=None)

    check.name = f'PRUNED_swd_{arguments.pruning_rate}_{arguments.a_min}_{arguments.a_max}'
    check.model.freeze(arguments.frozen_image_shape)
    torch.save(check.model, os.path.join(check.save_folder, check.name + '_model.pt'))


if __name__ == '__main__':
    main()
