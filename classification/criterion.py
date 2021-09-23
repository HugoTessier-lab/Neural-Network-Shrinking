import torch


def get_criterion(criterion):
    if criterion == 'crossentropy':
        criterion = torch.nn.CrossEntropyLoss()
        setattr(criterion, 'name', 'CrossEntropy')
        return criterion
    else:
        print('Invalid criterion type')
        raise ValueError
