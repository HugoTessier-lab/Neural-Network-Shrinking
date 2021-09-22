import torch
from utils.rmi import RMILoss


class SoftMaxMSELoss:
    def __init__(self):
        self.softmax = torch.nn.Softmax(dim=1)
        self.mse = torch.nn.MSELoss()

    def __call__(self, pred, target, num_classes=19):
        pred = self.softmax(pred)
        new_target = torch.stack([target == c for c in range(num_classes)], dim=1).to(target.device)
        return self.mse(pred, new_target.float())


def get_criterion(criterion):
    if criterion == 'crossentropy':
        return torch.nn.CrossEntropyLoss(ignore_index=19)
    elif criterion == 'mse':
        return SoftMaxMSELoss()
    elif criterion == 'rmi':
        return RMILoss(num_classes=19)
    else:
        print('Invalid criterion type')
        raise ValueError
