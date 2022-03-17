from utils import args, checkpoint, optimizer, scheduler, train, criterion, datasets, metrics, models
from pruning.pruning_criterion import get_pruning_criterion
from pruning.methods import get_pruning_method
import torch
import os


def main():
    arguments = args.parse_arguments()

    model = models.get_model(arguments).to(arguments.device)
    optim = optimizer.get_optimizer(arguments, model)
    sched = scheduler.get_scheduler(arguments, optim)
    check = checkpoint.Checkpoint(model, optim, sched, arguments.device, arguments.distributed,
                                  arguments.checkpoint_path, 'base')

    dataset = datasets.get_dataset(arguments)
    met = [metrics.get_metric(n) for n in arguments.metrics]
    crit = criterion.get_criterion(arguments.criterion)

    pruning_method = None
    if arguments.pruning_rate != 0 and arguments.pruning_method != 'none':
        pruning_criterion = get_pruning_criterion(arguments.pruning_criterion)
        pruning_method = get_pruning_method(arguments, model, dataset['train'], pruning_criterion)
        check.name += pruning_method.get_name('base')

    count = len(torch.cat([i.flatten() for i in check.model.parameters()]))
    print('Original count of parameters: ', count)

    train.train_model(name='Training', checkpoint=check, dataset=dataset,
                      epochs=arguments.epochs, criterion=crit, metrics=met,
                      output_path=arguments.results_path, debug=arguments.debug,
                      device=arguments.device,
                      pruning_method=pruning_method)

    if pruning_method:
        pruning_method.prune()

    new_count = len(torch.cat([i.flatten() for i in check.model.parameters()]))
    print('\nCount of parameters after pruning: ', new_count)
    print('Ratio: ', round((new_count / count) * 100, 2))

    if arguments.lrr:
        check.optimizer = optimizer.get_optimizer(arguments, check.model)
        check.scheduler = scheduler.get_scheduler(arguments, check.optimizer)
        check.name = 'LRR' + (pruning_method.get_name('pruned') if pruning_method is not None else '')
        train.train_model(name=f'LRR_{arguments.pruning_rate}', checkpoint=check, dataset=dataset,
                          epochs=arguments.epochs, criterion=crit, metrics=met,
                          output_path=arguments.results_path, debug=arguments.debug,
                          device=arguments.device,
                          pruning_method=None)

    check.name = 'PRUNED' + (pruning_method.get_name('pruned') if pruning_method is not None else '')
    if arguments.distributed:
        model = check.model.module
    else:
        model = check.model
    model.freeze(arguments.frozen_image_shape)
    torch.save(model, os.path.join(check.save_folder, check.name + '_model.pt'))


if __name__ == '__main__':
    main()
