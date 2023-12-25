import torch

def compute_losses_diff(opts, src_pcd0, src_pcd, pred_transforms, transform_gt, loss_type = 'mae', reduction = 'mean'):

    losses = {}
    num_iter = len(pred_transforms)

    # Compute losses
    gt_src_transformed = transform(transform_gt, src_pcd0)
    if loss_type == 'mse':
        # MSE loss to the groundtruth (does not take into account possible symmetries)
        criterion = torch.nn.MSELoss(reduction=reduction)
        for i in range(num_iter):
            pred_src_transformed = transform(pred_transforms[i], src_pcd)
            if reduction.lower() == 'mean':
                losses['mse_{}'.format(i)] = criterion(pred_src_transformed, gt_src_transformed)
            elif reduction.lower() == 'none':
                losses['mse_{}'.format(i)] = torch.mean(criterion(pred_src_transformed, gt_src_transformed),
                                                        dim=[-1, -2])
    elif loss_type == 'mae':
        # MSE loss to the groundtruth (does not take into account possible symmetries)
        criterion = torch.nn.L1Loss(reduction=reduction)
        for i in range(num_iter):
            pred_src_transformed = transform(pred_transforms[i], src_pcd)
            if reduction.lower() == 'mean':
                losses['mae_{}'.format(i)] = criterion(pred_src_transformed, gt_src_transformed)
            elif reduction.lower() == 'none':
                losses['mae_{}'.format(i)] = torch.mean(criterion(pred_src_transformed, gt_src_transformed), dim=[-1, -2])
    else:
        raise NotImplementedError

    discount_factor = 0.5  # Early iterations will be discounted
    total_losses = []
    for k in losses:
        discount = discount_factor ** (num_iter - int(k[k.rfind('_')+1:]) - 1)
        total_losses.append(losses[k] * discount)
    losses['total'] = torch.sum(torch.stack(total_losses), dim=0)

    return losses

def compute_losses(opts, src_pcd, pred_transforms, transform_gt, loss_type = 'mae', reduction = 'mean'):

    losses = {}
    num_iter = len(pred_transforms)

    # Compute losses
    gt_src_transformed = transform(transform_gt, src_pcd)
    if loss_type == 'mse':
        # MSE loss to the groundtruth (does not take into account possible symmetries)
        criterion = torch.nn.MSELoss(reduction=reduction)
        for i in range(num_iter):
            pred_src_transformed = transform(pred_transforms[i], src_pcd)
            if reduction.lower() == 'mean':
                losses['mse_{}'.format(i)] = criterion(pred_src_transformed, gt_src_transformed)
            elif reduction.lower() == 'none':
                losses['mse_{}'.format(i)] = torch.mean(criterion(pred_src_transformed, gt_src_transformed),
                                                        dim=[-1, -2])
    elif loss_type == 'mae':
        # MSE loss to the groundtruth (does not take into account possible symmetries)
        criterion = torch.nn.L1Loss(reduction=reduction)
        for i in range(num_iter):
            pred_src_transformed = transform(pred_transforms[i], src_pcd)
            if reduction.lower() == 'mean':
                losses['mae_{}'.format(i)] = criterion(pred_src_transformed, gt_src_transformed)
            elif reduction.lower() == 'none':
                losses['mae_{}'.format(i)] = torch.mean(criterion(pred_src_transformed, gt_src_transformed), dim=[-1, -2])
    else:
        raise NotImplementedError

    discount_factor = 0.5  # Early iterations will be discounted
    total_losses = []
    for k in losses:
        discount = discount_factor ** (num_iter - int(k[k.rfind('_')+1:]) - 1)
        total_losses.append(losses[k] * discount)
    losses['total'] = torch.sum(torch.stack(total_losses), dim=0)

    return losses

def transform(g, a, normals=None):
    R = g[..., :3, :3]  # (B, 3, 3)
    p = g[..., :3, 3]  # (B, 3)

    if len(g.size()) == len(a.size()):
        b = torch.matmul(a, R.transpose(-1, -2)) + p[..., None, :]
    else:
        raise NotImplementedError
        b = R.matmul(a.unsqueeze(-1)).squeeze(-1) + p  # No batch. Not checked

    if normals is not None:
        rotated_normals = normals @ R.transpose(-1, -2)
        return b, rotated_normals

    else:
        return b