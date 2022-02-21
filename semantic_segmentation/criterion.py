import torch
import torch.nn as nn
import torch.nn.functional as F

_euler_num = 2.718281828
_pi = 3.14159265
_ln_2_pi = 1.837877
_CLIP_MIN = 1e-6
_CLIP_MAX = 1.0
_POS_ALPHA = 5e-4
_IS_SUM = 1


def map_get_pairs(labels_4D, probs_4D, radius=3, is_combine=True):
    label_shape = labels_4D.size()
    h, w = label_shape[2], label_shape[3]
    new_h, new_w = h - (radius - 1), w - (radius - 1)
    la_ns = []
    pr_ns = []
    for y in range(0, radius, 1):
        for x in range(0, radius, 1):
            la_now = labels_4D[:, :, y:y + new_h, x:x + new_w]
            pr_now = probs_4D[:, :, y:y + new_h, x:x + new_w]
            la_ns.append(la_now)
            pr_ns.append(pr_now)

    if is_combine:
        pair_ns = la_ns + pr_ns
        p_vectors = torch.stack(pair_ns, dim=2)
        return p_vectors
    else:
        la_vectors = torch.stack(la_ns, dim=2)
        pr_vectors = torch.stack(pr_ns, dim=2)
        return la_vectors, pr_vectors


def map_get_pairs_region(labels_4D, probs_4D, radius=3, is_combine=0, num_classeses=21):
    kernel = torch.zeros([num_classeses, 1, radius, radius]).type_as(probs_4D)
    padding = radius // 2
    la_ns = []
    pr_ns = []
    for y in range(0, radius, 1):
        for x in range(0, radius, 1):
            kernel_now = kernel.clone()
            kernel_now[:, :, y, x] = 1.0
            la_now = F.conv2d(labels_4D, kernel_now, stride=radius, padding=padding, groups=num_classeses)
            pr_now = F.conv2d(probs_4D, kernel_now, stride=radius, padding=padding, groups=num_classeses)
            la_ns.append(la_now)
            pr_ns.append(pr_now)
    if is_combine:
        pair_ns = la_ns + pr_ns
        p_vectors = torch.stack(pair_ns, dim=2)
        return p_vectors
    else:
        la_vectors = torch.stack(la_ns, dim=2)
        pr_vectors = torch.stack(pr_ns, dim=2)
        return la_vectors, pr_vectors


def log_det_by_cholesky(matrix):
    chol = torch.linalg.cholesky(matrix)
    return 2.0 * torch.sum(torch.log(torch.diagonal(chol, dim1=-2, dim2=-1) + 1e-8), dim=-1)


def batch_cholesky_inverse(matrix):
    chol_low = torch.cholesky(matrix, upper=False)
    chol_low_inv = batch_low_tri_inv(chol_low)
    return torch.matmul(chol_low_inv.transpose(-2, -1), chol_low_inv)


def batch_low_tri_inv(L):
    n = L.shape[-1]
    invL = torch.zeros_like(L)
    for j in range(0, n):
        invL[..., j, j] = 1.0 / L[..., j, j]
        for i in range(j + 1, n):
            S = 0.0
            for k in range(0, i + 1):
                S = S - L[..., i, k] * invL[..., k, j].clone()
            invL[..., i, j] = S / L[..., i, i]
    return invL


def log_det_by_cholesky_test():
    a = torch.randn(1, 4, 4)
    a = torch.matmul(a, a.transpose(2, 1))
    print(a)
    res_1 = torch.logdet(torch.squeeze(a))
    res_2 = log_det_by_cholesky(a)
    print(res_1, res_2)


def batch_inv_test():
    a = torch.randn(1, 1, 4, 4)
    a = torch.matmul(a, a.transpose(-2, -1))
    print(a)
    res_1 = torch.inverse(a)
    res_2 = batch_cholesky_inverse(a)
    print(res_1, '\n', res_2)


def mean_var_test():
    x = torch.randn(3, 4)
    y = torch.randn(3, 4)
    x_mean = x.mean(dim=1, keepdim=True)
    x_sum = x.sum(dim=1, keepdim=True) / 2.0
    y_mean = y.mean(dim=1, keepdim=True)
    y_sum = y.sum(dim=1, keepdim=True) / 2.0
    x_var_1 = torch.matmul(x - x_mean, (x - x_mean).t())
    x_var_2 = torch.matmul(x, x.t()) - torch.matmul(x_sum, x_sum.t())
    xy_cov = torch.matmul(x - x_mean, (y - y_mean).t())
    xy_cov_1 = torch.matmul(x, y.t()) - x_sum.matmul(y_sum.t())
    print(x_var_1)
    print(x_var_2)
    print(xy_cov, '\n', xy_cov_1)


class RMILoss(nn.Module):

    def __init__(self,
                 num_classes=21,
                 rmi_radius=3,
                 rmi_pool_way=0,
                 rmi_pool_size=3,
                 rmi_pool_stride=3,
                 loss_weight_lambda=0.5,
                 lambda_way=1):
        super(RMILoss, self).__init__()
        self.num_classes = num_classes
        self.rmi_radius = rmi_radius
        self.rmi_pool_way = rmi_pool_way
        self.rmi_pool_size = rmi_pool_size
        self.rmi_pool_stride = rmi_pool_stride
        self.weight_lambda = loss_weight_lambda
        self.lambda_way = lambda_way
        self.half_d = self.rmi_radius * self.rmi_radius
        self.d = 2 * self.half_d
        self.kernel_padding = self.rmi_pool_size // 2
        self.ignore_index = 255
        self.name = 'RMI'

    def forward(self, logits_4D, labels_4D):
        loss = self.forward_sigmoid(logits_4D, labels_4D)
        return loss

    def forward_softmax_sigmoid(self, logits_4D, labels_4D):

        normal_loss = F.cross_entropy(input=logits_4D,
                                      target=labels_4D.long(),
                                      ignore_index=self.ignore_index,
                                      reduction='mean')
        label_mask_3D = labels_4D < self.num_classes
        valid_onehot_labels_4D = F.one_hot(labels_4D.long() * label_mask_3D.long(),
                                           num_classes=self.num_classes).float()
        label_mask_3D = label_mask_3D.float()
        valid_onehot_labels_4D = valid_onehot_labels_4D * label_mask_3D.unsqueeze(dim=3)
        valid_onehot_labels_4D = valid_onehot_labels_4D.permute(0, 3, 1, 2).requires_grad_(False)
        probs_4D = F.sigmoid(logits_4D) * label_mask_3D.unsqueeze(dim=1)
        probs_4D = probs_4D.clamp(min=_CLIP_MIN, max=_CLIP_MAX)
        rmi_loss = self.rmi_lower_bound(valid_onehot_labels_4D, probs_4D)
        final_loss = (self.weight_lambda * normal_loss + rmi_loss * (1 - self.weight_lambda) if self.lambda_way
                      else normal_loss + rmi_loss * self.weight_lambda)

        return final_loss

    def forward_sigmoid(self, logits_4D, labels_4D):
        label_mask_3D = labels_4D < self.num_classes
        valid_onehot_labels_4D = F.one_hot(labels_4D.long() * label_mask_3D.long(),
                                           num_classes=self.num_classes).float()
        label_mask_3D = label_mask_3D.float()
        label_mask_flat = label_mask_3D.view([-1, ])
        valid_onehot_labels_4D = valid_onehot_labels_4D * label_mask_3D.unsqueeze(dim=3)
        valid_onehot_labels_4D.requires_grad_(False)
        valid_onehot_label_flat = valid_onehot_labels_4D.view([-1, self.num_classes]).requires_grad_(False)
        logits_flat = logits_4D.permute(0, 2, 3, 1).contiguous().view([-1, self.num_classes])
        valid_pixels = torch.sum(label_mask_flat)
        binary_loss = F.binary_cross_entropy_with_logits(logits_flat,
                                                         target=valid_onehot_label_flat,
                                                         weight=label_mask_flat.unsqueeze(dim=1),
                                                         reduction='sum')
        bce_loss = torch.div(binary_loss, valid_pixels + 1.0)
        probs_4D = logits_4D.sigmoid() * label_mask_3D.unsqueeze(dim=1) + _CLIP_MIN
        valid_onehot_labels_4D = valid_onehot_labels_4D.permute(0, 3, 1, 2).requires_grad_(False)
        rmi_loss = self.rmi_lower_bound(valid_onehot_labels_4D, probs_4D)
        final_loss = (self.weight_lambda * bce_loss + rmi_loss * (1 - self.weight_lambda) if self.lambda_way
                      else bce_loss + rmi_loss * self.weight_lambda)

        return final_loss

    def rmi_lower_bound(self, labels_4D, probs_4D):
        p, s = self.rmi_pool_size, self.rmi_pool_stride
        if self.rmi_pool_stride > 1:
            if self.rmi_pool_way == 0:
                labels_4D = F.max_pool2d(labels_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
                probs_4D = F.max_pool2d(probs_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
            elif self.rmi_pool_way == 1:
                labels_4D = F.avg_pool2d(labels_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
                probs_4D = F.avg_pool2d(probs_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
            elif self.rmi_pool_way == 2:
                shape = labels_4D.size()
                new_h, new_w = shape[2] // s, shape[3] // s
                labels_4D = F.interpolate(labels_4D, size=(new_h, new_w), mode='nearest')
                probs_4D = F.interpolate(probs_4D, size=(new_h, new_w), mode='bilinear', align_corners=True)
            else:
                raise NotImplementedError("Pool way of RMI is not defined!")
        label_shape = labels_4D.size()
        n, c = label_shape[0], label_shape[1]
        la_vectors, pr_vectors = map_get_pairs(labels_4D, probs_4D, radius=self.rmi_radius, is_combine=0)
        la_vectors = la_vectors.view([n, c, self.half_d, -1]).type(torch.cuda.DoubleTensor).requires_grad_(False)
        pr_vectors = pr_vectors.view([n, c, self.half_d, -1]).type(torch.cuda.DoubleTensor)
        diag_matrix = torch.eye(self.half_d).unsqueeze(dim=0).unsqueeze(dim=0)
        la_vectors = la_vectors - la_vectors.mean(dim=3, keepdim=True)
        la_cov = torch.matmul(la_vectors, la_vectors.transpose(2, 3))
        pr_vectors = pr_vectors - pr_vectors.mean(dim=3, keepdim=True)
        pr_cov = torch.matmul(pr_vectors, pr_vectors.transpose(2, 3))
        pr_cov_inv = torch.inverse(pr_cov + diag_matrix.type_as(pr_cov) * _POS_ALPHA)
        la_pr_cov = torch.matmul(la_vectors, pr_vectors.transpose(2, 3))
        appro_var = la_cov - torch.matmul(la_pr_cov.matmul(pr_cov_inv), la_pr_cov.transpose(-2, -1))
        rmi_now = 0.5 * log_det_by_cholesky(appro_var + diag_matrix.type_as(appro_var) * _POS_ALPHA)
        rmi_per_class = rmi_now.view([-1, self.num_classes]).mean(dim=0).float()
        rmi_per_class = torch.div(rmi_per_class, float(self.half_d))
        rmi_loss = torch.sum(rmi_per_class) if _IS_SUM else torch.mean(rmi_per_class)
        return rmi_loss


def get_criterion(criterion):
    if criterion == 'crossentropy':
        criterion = torch.nn.CrossEntropyLoss(ignore_index=19)
        setattr(criterion, 'name', 'CrossEntropy')
        return criterion
    elif criterion == 'rmi':
        return RMILoss(num_classes=19)
    else:
        print('Invalid criterion type')
        raise ValueError
