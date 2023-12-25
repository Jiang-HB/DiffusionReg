import os, numpy as np, random, torch, argparse, pdb
from utils.options import opts
from utils.diffusion_scheduler import DiffusionScheduler
from utils.get_lr_scheduler import get_lr_scheduler
from tqdm import tqdm
from collections import defaultdict
from utils.se_math import se3
from datasets.get_dataset import get_dataset
from utils.losses import compute_losses, compute_losses_diff

np.random.seed(opts.seed)
random.seed(opts.seed)
torch.manual_seed(opts.seed)
torch.backends.cudnn.benchmark=True
torch.backends.cudnn.enabled = False

def init_opts(opts):
    opts.is_debug = False
    opts.model_nm = "DiffusionReg"
    opts.is_test = False
    opts.is_normal = True
    opts.n_cores = 6
    opts.schedule_type = ["linear", "cosine"][1]

    # dataset config
    if opts.db_nm == "tudl":
        opts.n_vids = 3
        opts.n_epoches, opts.n_start_epoches, opts.batch_size = 20, 0, 32
        opts.vid_infos = ["000001", "000002", "000003"]
    else:
        raise NotImplementedError

    # diffusion configuration
    opts.n_diff_steps = 200
    opts.beta_1 = 1e-4
    opts.beta_T = 0.05
    opts.sigma_r = 0.05
    opts.sigma_t = 0.03
    diffusion_str = f"diffusion_{opts.n_diff_steps}_{opts.beta_1:.5f}_{opts.beta_T:.2f}_{opts.sigma_r:.2f}_{opts.sigma_t:.2f}"

    opts.results_dir = f"./results/{opts.model_nm}-{opts.net_type}-{opts.db_nm}-{diffusion_str}-nvids{opts.n_vids}_{opts.schedule_type}"
    os.makedirs(opts.results_dir, exist_ok=True)
    print(opts.results_dir)
    return opts

def main(opts):

    opts = init_opts(opts)

    ## model setting
    from modules.DCP.dcp import DCP
    opts.vs = DiffusionScheduler(opts)
    surrogate_model = DCP(opts)

    if torch.cuda.device_count() > 1:
        surrogate_model = torch.nn.DataParallel(surrogate_model, range(torch.cuda.device_count()))
    surrogate_model = surrogate_model.to(opts.device)

    train_loader, train_db = get_dataset(opts, db_nm=opts.db_nm, cls_nm=opts.vid_infos, partition="train",
                                         batch_size=opts.batch_size, shuffle=True, drop_last=True, n_cores=opts.n_cores)
    optimizer = torch.optim.Adam(surrogate_model.parameters(), lr=opts.lr, betas=(0.9, 0.999))
    scheduler = get_lr_scheduler(opts, optimizer)
    cal = lambda x: np.mean(x).item()

    ## training
    for epoch_idx in range(opts.n_epoches):

        # train
        surrogate_model.train()

        rcd = defaultdict(list)
        for i, data in enumerate(tqdm(train_loader, 0)):

            data = {k: v.to(opts.device) for k, v in data.items()}

            # model prediction
            X, X_normal = data["src_pcd"], data["src_pcd_normal"]  # [B, N, 3]
            Y, Y_normal = data["model_pcd"], data["model_pcd_normal"]  # [B, M, 3]
            Rs_gt, ts_gt = data['transform_gt'][:, :3, :3], data["transform_gt"][:, :3, 3]  # [B, 3, 3], [B, 3]
            B = Rs_gt.shape[0]

            ### SE(3) diffusion process
            H_0 = torch.eye(4)[None].expand(B, -1, -1).to(opts.device)
            H_0[:, :3, :3], H_0[:, :3, 3] = Rs_gt, ts_gt
            H_T = torch.eye(4)[None].expand(B, -1, -1).to(opts.device)

            taus = opts.vs.uniform_sample_t(B)
            alpha_bars = opts.vs.alpha_bars[taus].to(opts.device)[:, None]  # [B, 1]
            H_t = se3.exp((1. - torch.sqrt(alpha_bars)) * se3.log(H_T @ torch.inverse(H_0))) @ H_0

            ### add noise
            scale = torch.cat([torch.ones(3) * opts.sigma_r, torch.ones(3) * opts.sigma_t])[None].to(opts.device)  # [1, 6]
            noise = torch.sqrt(1. - alpha_bars) * scale * torch.randn(B, 6).to(opts.device)  # [B, 6]
            H_noise = se3.exp(noise)
            H_t_noise = H_noise @ H_t  # [B, 4, 4]

            T_t_R = H_t_noise[:, :3, :3]  # [B, 3, 3]
            T_t_t = H_t_noise[:, :3, 3]  # [B, 3]

            X_t = (T_t_R @ X.transpose(2, 1) + T_t_t.unsqueeze(-1)).transpose(2, 1)  # [B, N, 3]
            X_normal_t = (T_t_R @ X_normal.transpose(2, 1)).transpose(2, 1)          # [B, N, 3]

            transform_gt = torch.eye(4)[None].expand(B, -1, -1).to(opts.device)
            transform_gt[:, :3] = data['transform_gt']
            input = {
                "src_pcd": X_t,
                "src_pcd_normal": X_normal_t,
                "model_pcd": Y,
                "model_pcd_normal": Y_normal,
            }
            Rs_pred_rot, ts_pred_rot = surrogate_model.forward(input)
            pred_transforms = torch.cat([Rs_pred_rot, ts_pred_rot.unsqueeze(-1)], dim=2)  # [B, 3, 4]
            train_losses_diff = compute_losses_diff(opts, X, X_t, [pred_transforms], data['transform_gt'], loss_type="mae", reduction='mean')
            loss = train_losses_diff['total']

            # original loss
            input = {
                "src_pcd": X,
                "src_pcd_normal": X_normal,
                "model_pcd": Y,
                "model_pcd_normal": Y_normal,
            }
            Rs_pred_rot1, ts_pred_rot1 = surrogate_model.forward(input)
            pred_transforms1 = torch.cat([Rs_pred_rot1, ts_pred_rot1.unsqueeze(-1)], dim=2)  # [B, 3, 4]
            train_losses_origin = compute_losses(opts, X, [pred_transforms1], data['transform_gt'], loss_type="mae", reduction='mean')
            loss += train_losses_origin["total"]
            rcd["losses"].append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("=== Train. Epoch [%d], losses: %1.3f ===" % (epoch_idx, cal(rcd["losses"])))

            if i > 0 and i % 200 == 0 and not opts.is_debug:
                print("Save model. %s" % ('%s/model_epoch%d.pth' % (opts.results_dir, epoch_idx)))
                torch.save(surrogate_model.state_dict(), '%s/model_epoch%d.pth' % (opts.results_dir, epoch_idx))

        print(opts.results_dir)

        # save model
        if not opts.is_debug:
            print("Save model. %s" % ('%s/model_epoch%d.pth' % (opts.results_dir, epoch_idx)))
            torch.save(surrogate_model.state_dict(), '%s/model_epoch%d.pth' % (opts.results_dir, epoch_idx))
        else:
            print("Debug. Not save model.")

        scheduler.step()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--net_type', default="DiffusionDCP", type=str, choices=['DiffusionDCP'])
    parser.add_argument('--db_nm', default="tudl", type=str, choices=["tudl"])
    args = parser.parse_args()

    opts.net_type = args.net_type
    opts.db_nm = args.db_nm
    main(opts)