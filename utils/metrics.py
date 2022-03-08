import torch


class MIOU:
    @staticmethod
    def IoU(x, y, smooth=1):
        intersection = (x * y).abs().sum(dim=[1, 2])
        union = torch.sum(y.abs() + x.abs(), dim=[1, 2]) - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou

    @staticmethod
    def get_mask(target, num_classes=19):
        mask = (target >= 0) & (target < num_classes)
        return mask.float()

    def __init__(self):
        self.name = 'miou'

    def __call__(self, output, target):
        l = list()
        mask = self.get_mask(target)
        transformed_output = output.permute(0, 2, 3, 1).argmax(dim=3)
        for c in range(output.shape[1]):
            x = (transformed_output == c).float() * mask
            y = (target == c).float()
            l.append(self.IoU(x, y))
        return torch.sum(torch.mean(torch.stack(l).permute(1, 0), dim=1)).item()


class Top5Accuracy:
    def __init__(self):
        self.name = 'Top5Accuracy'

    def __call__(self, output, target):
        result = 0
        for o, t in zip(output, target):
            result += int(t in torch.argsort(o, descending=True)[:5])
        return result


class Top1Accuracy:
    def __init__(self):
        self.name = 'Top1Accuracy'

    def __call__(self, output, target):
        pred = output.argmax(dim=1, keepdim=True)
        return pred.eq(target.view_as(pred)).sum().item()


def get_metric(name):
    if name == 'miou':
        return MIOU()
    elif name == 'top1':
        return Top1Accuracy()
    elif name == 'top5':
        return Top5Accuracy()
    else:
        print('Wrong metric type.')
        raise ValueError
