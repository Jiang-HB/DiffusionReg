import numpy as np

def evaluate_Rt(R_gt, t_gt, R_pred, t_pred):
    t_pred = t_pred.flatten()
    t_gt = t_gt.flatten()
    R_err = np.arccos(np.clip((np.trace(R_pred.T @ R_gt) - 1.) / 2., -1., 1.0))
    t_err = np.sum((t_pred - t_gt) ** 2)

    return R_err, t_err

def mAP(Rs_pred, ts_pred, Rs_gt, ts_gt):
    """
    Rs_pred: [B, 3, 3]
    Rs_gt: [B, 3, 3]
    ts_pred: [B, 3]
    ts_gt: [B, 3]
    """
    Rs_err, ts_err = [], []
    for idx in range(len(Rs_pred)):
        R_pred, t_pred = Rs_pred[idx], ts_pred[idx]
        R_gt, t_gt = Rs_gt[idx], ts_gt[idx]
        R_err, t_err = evaluate_Rt(R_gt, t_gt, R_pred, t_pred)
        Rs_err.append(R_err)
        ts_err.append(t_err)
    Rs_err, ts_err = np.asarray(Rs_err), np.asarray(ts_err)
    Rs_err = Rs_err * 180. / np.pi

    R_ths = np.array([0., 5., 10., 20.])
    t_ths = np.array([0., 1., 2., 5.])
    R_acc_hist, _ = np.histogram(Rs_err, R_ths)
    t_acc_hist, _ = np.histogram(ts_err, t_ths)
    num_pair = float(len(Rs_err))
    R_acc_hist = R_acc_hist.astype(float) / num_pair
    t_acc_hist = t_acc_hist.astype(float) / num_pair
    R_acc = np.cumsum(R_acc_hist)
    t_acc = np.cumsum(t_acc_hist)

    return R_acc, t_acc

