import torch
import sys
import os


def train_model(checkpoint, dataset, epochs, criterion, metrics, output_path, debug):
    e = 0
    while e < epochs:
        e = checkpoint.store_model(e)
        if e >= epochs:
            break
        print(f'\nEpoch {e + 1}/{epochs}')

        results = [0 for _ in range(len(metrics))]
        global_loss = 0
        checkpoint.model.train()
        for i, (data, target) in enumerate(dataset['train']):
            if debug:
                if i != 0:
                    break
            data, target = data.cuda(), target.cuda()
            checkpoint.optimizer.zero_grad()
            output = checkpoint.model(data)
            loss = criterion(output, target.long())
            loss.backward()
            checkpoint.optimizer.step()
            for k, m in enumerate(metrics):
                results[k] += m(output, target)
            global_loss += loss.item()

            message = f'\rTrain ({i + 1}/{len(dataset["train"])}) -> '
            message += f'{criterion.name} loss = {round(float(global_loss) / (i + 1), 3)}, '
            for k, m in enumerate(metrics):
                message += f'{m.name} = {round(float(results[k]) / (i + 1), 3)}\t'
            message += '           '
            sys.stdout.write(message)

        print()
        checkpoint.model.eval()
        with torch.no_grad():
            results = [0 for _ in range(len(metrics))]
            global_loss = 0
            for i, (data, target) in enumerate(dataset['test']):
                if debug:
                    if i != 0:
                        break
                data, target = data.cuda(), target.cuda()

                output = checkpoint.model(data)
                loss = criterion(output, target.long())
                for k, m in enumerate(metrics):
                    results[k] += m(output, target)
                global_loss += loss.item()

                message = f'\rTest ({i + 1}/{len(dataset["test"])}) -> '
                message += f'{criterion.name} loss = {round(float(global_loss) / (i + 1), 3)}, '
                for k, m in enumerate(metrics):
                    message += f'{m.name} = {round(float(results[k]) / (i + 1), 3)}\t'
                message += '           '
                sys.stdout.write(message)

        with open(os.path.join(output_path, 'results.txt'), 'a') as f:
            message = f'Epoch {e + 1}/{epochs} -> '
            message += f'{criterion.name} loss = {float(global_loss) / len(dataset["test"])}, '
            for k, m in enumerate(metrics):
                message += f'{m.name} = {float(results[k]) / len(dataset["test"])}\t'
            message += '\n'
            f.write(message)

        e += 1
        checkpoint.scheduler.step()

    checkpoint.store_model(epochs)
