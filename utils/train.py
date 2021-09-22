import torch
import sys
import os
from matplotlib import pyplot as plt
import pickle
from utils.criterion import get_criterion
from torch.nn import functional as F


def save_images(target, output, path, epoch, print_images):
    if print_images:
        plt.clf()
        plt.close('all')
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.suptitle(f'Epoch {epoch}')
        ax1.imshow(target)
        ax1.set_title('Target')
        ax1.axis('off')
        ax2.imshow(output.argmax(0))
        ax2.set_title('Output')
        ax2.axis('off')
        plt.savefig(path)


def IoU(x, y, smooth=1):
    intersection = (x * y).abs().sum(dim=[1, 2])
    union = torch.sum(y.abs() + x.abs(), dim=[1, 2]) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou


def get_mask(target, num_classes=19):
    mask = (target >= 0) & (target < num_classes)
    return mask.float()


def mIoU(output, target):
    l = list()
    mask = get_mask(target)
    transformed_output = output.permute(0, 2, 3, 1).argmax(dim=3)
    for c in range(output.shape[1]):
        x = (transformed_output == c).float() * mask
        y = (target == c).float()
        l.append(IoU(x, y))
    return torch.mean(torch.stack(l)).item()


def train_model(checkpoint, dataset, epochs, output_path, debug, criterion, swd=None, print_images=False):
    criterion = get_criterion(criterion)

    e = 0
    while e < epochs:
        e = checkpoint.store_model(output_path, e)
        if e >= epochs:
            break
        print(f'\nEpoch {e + 1}/{epochs}')

        miou = 0
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
            if swd:
                loss += swd(checkpoint.model)
            loss.backward()
            checkpoint.optimizer.step()
            miou += mIoU(output, target)
            global_loss += loss.item()

            sys.stdout.write(f'\rTrain ({i + 1}/{len(dataset["train"])}) -> '
                             f'loss = {round(float(global_loss) / (i + 1), 3)}, '
                             f'mIoU = {round(float(miou) / (i + 1), 3)}               ')

        print()
        checkpoint.model.eval()
        with torch.no_grad():
            miou = 0
            global_loss = 0
            for i, (data, target) in enumerate(dataset['test']):
                if debug:
                    if i != 0:
                        break
                data, target = data.cuda(), target.cuda()

                output = checkpoint.model(data)
                loss = criterion(output, target.long())
                miou += mIoU(output, target)
                global_loss += loss.item()

                sys.stdout.write(
                    f'\rTest ({i + 1}/{len(dataset["test"])}) -> '
                    f'loss = {round(float(global_loss) / (i + 1), 3)}, '
                    f'mIoU = {round(float(miou) / (i + 1), 3)}               ')
                if i == 0:
                    save_images(target[0].detach().cpu().numpy(), output[0].detach().cpu().numpy(),
                                os.path.join(output_path, f'epoch{e + 1}.png'), e + 1, print_images)

        with open(os.path.join(output_path, 'results.txt'), 'a') as f:
            f.write(f'Epoch {e + 1}/{epochs} : loss = {float(global_loss) / len(dataset["test"])}, '
                    f'mIoU = {float(miou) / len(dataset["test"])}\n')

        e += 1
        checkpoint.scheduler.step()

    checkpoint.store_model(output_path, epochs)

    print('\nOne pass over the train set to reset batchnorms')
    miou = 0
    global_loss = 0
    checkpoint.model.train()
    for i, (data, target) in enumerate(dataset['reset']):
        if debug:
            if i != 0:
                break
        data, target = data.cuda(), target.cuda()
        checkpoint.optimizer.zero_grad()
        output = checkpoint.model(data)
        loss = criterion(output, target.long())
        miou += mIoU(output, target)
        global_loss += loss.item()

        sys.stdout.write(f'\rTrain ({i + 1}/{len(dataset["reset"])}) -> '
                         f'loss = {round(float(global_loss) / (i + 1), 3)}, '
                         f'mIoU = {round(float(miou) / (i + 1), 3)}               ')
    print('\nFinal test')
    checkpoint.model.eval()
    with torch.no_grad():
        miou = 0
        global_loss = 0
        for i, (data, target) in enumerate(dataset['test']):
            if debug:
                if i != 0:
                    break
            data, target = data.cuda(), target.cuda()

            output = checkpoint.model(data)
            loss = criterion(output, target.long())
            miou += mIoU(output, target)
            global_loss += loss.item()

            sys.stdout.write(
                f'\rTest ({i + 1}/{len(dataset["test"])}) -> '
                f'loss = {round(float(global_loss) / (i + 1), 3)}, '
                f'mIoU = {round(float(miou) / (i + 1), 3)}               ')
            if i == 0:
                save_images(target[0].detach().cpu().numpy(), output[0].detach().cpu().numpy(),
                            os.path.join(output_path, f'epoch{e + 1}.png'), e + 1, print_images)

    with open(os.path.join(output_path, 'results.txt'), 'a') as f:
        f.write(f'Final results : loss = {float(global_loss) / len(dataset["test"])}, '
                f'mIoU = {float(miou) / len(dataset["test"])}\n')

    if swd:
        print('\nPruning model')
        checkpoint.model = swd.prune(checkpoint.model)

        checkpoint.model.eval()
        with torch.no_grad():
            miou = 0
            global_loss = 0
            for i, (data, target) in enumerate(dataset['test']):
                if debug:
                    if i != 0:
                        break
                data, target = data.cuda(), target.cuda()

                output = checkpoint.model(data)
                loss = F.cross_entropy(output, target.long())
                miou += mIoU(output, target)
                global_loss += loss.item()

                sys.stdout.write(
                    f'\rTest ({i + 1}/{len(dataset["test"])}) -> '
                    f'loss = {round(float(global_loss) / (i + 1), 3)}, '
                    f'mIoU = {round(float(miou) / (i + 1), 3)}               ')
                if i == 0:
                    save_images(target[0].detach().cpu().numpy(), output[0].detach().cpu().numpy(),
                                os.path.join(output_path, f'pruned.png'), e + 1, print_images)

        with open(os.path.join(output_path, 'pruned_model.chk'), 'wb') as f:
            pickle.dump(checkpoint.model.state_dict(), f)
        with open(os.path.join(output_path, 'results.txt'), 'a') as f:
            f.write(f'Post-Pruning : loss = {float(global_loss) / len(dataset["test"])}, '
                    f'mIoU = {float(miou) / len(dataset["test"])}\n')
