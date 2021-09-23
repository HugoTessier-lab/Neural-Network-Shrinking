import pickle
import torch
import os


class Checkpoint:
    def __init__(self, model, optimizer, scheduler, device, distributed, save_folder):
        self.model = model.to(device)
        if distributed:
            self.model = torch.nn.DataParallel(model)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_folder = save_folder

    def save_model(self, path, epoch):
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model.state_dict(),
                         'epoch': epoch,
                         'optimizer': self.optimizer.state_dict()}, f)

    def store_model(self, epoch):
        if not os.path.isdir(self.save_folder):
            try:
                os.mkdir(self.save_folder)
            except OSError:
                print(f'Failed to create the folder {self.save_folder}')
            else:
                print(f'Created folder {self.save_folder}')
        path = os.path.join(self.save_folder, 'model.chk')
        if not os.path.isfile(path):
            self.save_model(path, epoch)
            return epoch
        else:
            with open(path, 'rb') as f:
                checkpoint = pickle.load(f)
            loaded_epoch = checkpoint['epoch']
            if epoch >= loaded_epoch:
                self.save_model(path, epoch)
                return epoch
            elif epoch < loaded_epoch:
                self.model.load_state_dict(checkpoint['model'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.last_epoch = loaded_epoch
                return loaded_epoch
            else:
                raise ValueError
