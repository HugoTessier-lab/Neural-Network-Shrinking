from utils import args, checkpoint, optimizer, scheduler, train, criterion, datasets, metrics, models
from pruning.pruning_criterion import get_pruning_criterion
from pruning.methods import get_pruning_method
import torch
import os
import numpy as np


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

    print('\n\nTraining')
    train.train_model(name='Training', checkpoint=check, dataset=dataset,
                      epochs=arguments.epochs, criterion=crit, metrics=met,
                      output_path=arguments.results_path, debug=arguments.debug,
                      device=arguments.device,
                      pruning_method=pruning_method)

    pruning_iterations = arguments.pruning_iterations - arguments.lrr

    if arguments.pruning_method == "liu2017" and pruning_iterations != 1:
        rates = np.linspace(0, arguments.pruning_rate, arguments.pruning_iterations + 1)[1:]
        for i, r in enumerate(rates if not arguments.lrr else rates[:-1]):
            pruning_method.pruning_rate = r
            pruning_method.mask_model()
            name = 'FT' + pruning_method.get_name(f'{i + 1}_{arguments.pruning_iterations}')
            print(f'\n\n{name}')
            check.name = name + f'_final_pruning_rate_{arguments.pruning_rate}_step({i + 1}_{len(rates)})'
            train.train_model(name=name, checkpoint=check, dataset=dataset,
                              epochs=arguments.fine_tuning_epochs if i + 1 != len(rates) and not arguments.lrr
                              else arguments.fine_tuning_epochs + arguments.additional_final_epochs,
                              criterion=crit, metrics=met,
                              output_path=arguments.results_path, debug=arguments.debug,
                              device=arguments.device,
                              pruning_method=pruning_method,
                              freeze_lr=True)
            pruning_method.remove()
        pruning_method.pruning_rate = rates[-1]

    if arguments.lrr:
        pruning_method.mask_model()
        check.optimizer = optimizer.get_optimizer(arguments, check.model)
        check.scheduler = scheduler.get_scheduler(arguments, check.optimizer)
        check.name = 'LRR' + (pruning_method.get_name('pruned') if pruning_method is not None else '')
        print(f'\n\nLRR_{arguments.pruning_rate}')
        train.train_model(name=f'LRR_{arguments.pruning_rate}', checkpoint=check, dataset=dataset,
                          epochs=arguments.epochs, criterion=crit, metrics=met,
                          output_path=arguments.results_path, debug=arguments.debug,
                          device=arguments.device,
                          pruning_method=None)
        pruning_method.remove()

    if pruning_method:
        check.name = 'PRUNED' + pruning_method.get_name('pruned') + ('_lrr' if arguments.lrr else '')
        train.test_model(name='Before final pruning', checkpoint=check, dataset=dataset, epochs=arguments.epochs,
                         criterion=crit, metrics=met, output_path=arguments.results_path, debug=arguments.debug,
                         device=arguments.device)
        pruning_method.prune()
        train.test_model(name='After final pruning', checkpoint=check, dataset=dataset, epochs=arguments.epochs,
                         criterion=crit, metrics=met, output_path=arguments.results_path, debug=arguments.debug,
                         device=arguments.device)
        new_count = len(torch.cat([i.flatten() for i in check.model.parameters()]))
        print('\nCount of parameters after pruning: ', new_count)
        print('Ratio: ', round((new_count / count) * 100, 2))
    else:
        check.name = 'BASELINE'
    if arguments.distributed:
        model = check.model.module
    else:
        model = check.model
    torch.save(model, os.path.join(check.save_folder, check.name + '_model.pt'))


if __name__ == '__main__':
    main()
