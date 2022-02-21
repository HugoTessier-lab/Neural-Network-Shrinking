import torch
import sys
import os


def _save_results(name, criterion, dataset, e, epochs, global_loss, metrics, output_path, results):
    with open(os.path.join(output_path, 'results.txt'), 'a') as f:
        message = f'{name}: epoch {e + 1}/{epochs} -> '
        message += f'{criterion.name} loss = {float(global_loss) / (len(dataset["test"]) * dataset["test"].batch_size)}, '
        for k, m in enumerate(metrics):
            message += f'{m.name} = {float(results[k]) / (len(dataset["test"]) * dataset["test"].batch_size)}\t'
        message += '\n'
        f.write(message)


def _test(checkpoint, criterion, dataset, debug, device, metrics):
    checkpoint.model.eval()
    with torch.no_grad():
        results = [0 for _ in range(len(metrics))]
        global_loss = 0
        for i, (data, target) in enumerate(dataset['test']):
            if debug:
                if i != 0:
                    break
            data, target = data.to(device), target.to(device)

            output = checkpoint.model(data)
            loss = criterion(output, target.long())

            for k, m in enumerate(metrics):
                results[k] += m(output, target)
            global_loss += loss.item()

            message = f'\rTest ({i + 1}/{len(dataset["test"])}) -> '
            message += f'{criterion.name} loss = {round(float(global_loss) / ((i + 1) * dataset["test"].batch_size), 3)}, '
            for k, m in enumerate(metrics):
                message += f'{m.name} = {round(float(results[k]) / ((i + 1) * dataset["test"].batch_size), 3)}\t'
            message += '           '
            sys.stdout.write(message)
    return global_loss, results


def _train(checkpoint, criterion, dataset, debug, device, metrics, swd):
    results = [0 for _ in range(len(metrics))]
    global_loss = 0
    checkpoint.model.train()
    for i, (data, target) in enumerate(dataset['train']):
        if debug:
            if i != 0:
                break
        data, target = data.to(device), target.to(device)
        checkpoint.optimizer.zero_grad()

        output = checkpoint.model(data)
        loss = criterion(output, target.long())
        loss.backward()
        if swd:
            swd.step()
        checkpoint.optimizer.step()

        for k, m in enumerate(metrics):
            results[k] += m(output, target)
        global_loss += loss.item()

        message = f'\rTrain ({i + 1}/{len(dataset["train"])}) -> '
        message += f'{criterion.name} loss = {round(float(global_loss) / ((i + 1) * dataset["train"].batch_size), 3)}, '
        for k, m in enumerate(metrics):
            message += f'{m.name} = {round(float(results[k]) / ((i + 1) * dataset["train"].batch_size), 3)}\t'
        message += '           '
        sys.stdout.write(message)


def train_model(name, checkpoint, dataset, epochs, criterion, metrics, output_path, debug, device, swd):
    e = 0
    while e < epochs:
        e = checkpoint.store_model(e)
        if e >= epochs:
            break
        e += 1
        print(f'\nEpoch {e}/{epochs}')

        _train(checkpoint, criterion, dataset, debug, device, metrics, swd)
        print()
        global_loss, results = _test(checkpoint, criterion, dataset, debug, device, metrics)

        _save_results(name, criterion, dataset, e, epochs, global_loss, metrics, output_path, results)

        checkpoint.scheduler.step()
    checkpoint.store_model(e)
