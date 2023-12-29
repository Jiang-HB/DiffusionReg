import numpy as np, random, torch, pdb
from utils.options import opts
from datasets.get_dataset import get_dataset
from collections import OrderedDict
from utils.commons import save_data, load_data
from collections import defaultdict
from utils.criterion import mAP
from tqdm import tqdm
from utils.se_math import se3
from utils.diffusion_scheduler import DiffusionScheduler

opts.seed = 1234
np.random.seed(opts.seed)
random.seed(opts.seed)
torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def init_opts(opts):
    opts.is_debug = False
    opts.is_test = True
    opts.schedule_type = ["linear", "cosine"][1]

    opts.is_save_res = True
    opts.n_diff_steps = 5
    opts.beta_1 = 0.2
    opts.beta_T = 0.8
    opts.sigma_r = 0.1
    opts.sigma_t = 0.01
    opts.is_add_noise = True

    return opts


def get_model(opts):
    opts.model_type = "DiffusionDCP"
    opts.model_path = "./results/DiffusionReg-DiffusionDCP-tudl-diffusion_200_0.00010_0.05_0.05_0.03-nvids3_cosine/model_epoch19.pth"
    opts.save_path = f"./results/DiffusionReg-DiffusionDCP-tudl-diffusion_200_0.00010_0.05_0.05_0.03-nvids3_cosine/eval_results/model_epoch19_T{opts.n_diff_steps}_{opts.schedule_type}_{opts.db_nm}_{opts.vid_infos[0]}_noise{opts.is_add_noise}_v1.pth"

    # model config
    from modules.DCP.dcp import DCP
    surrogate_model = DCP(opts)
    opts.vs = DiffusionScheduler(opts)

    try:
        surrogate_model.load_state_dict(OrderedDict({k[7:]: v for k, v in torch.load(opts.model_path, map_location=opts.device).items()}))
    except:
        surrogate_model.load_state_dict(OrderedDict({k: v for k, v in torch.load(opts.model_path, map_location=opts.device).items()}))
    surrogate_model = surrogate_model.to(opts.device)
    surrogate_model.eval()

    print(opts.save_path)
    return surrogate_model

def main(opts):
    ## initial setting
    opts = init_opts(opts)
    surrogate_model = get_model(opts)
    test_loader, test_db = get_dataset(opts, db_nm=opts.db_nm, cls_nm=opts.vid_infos, partition="test", batch_size=1, shuffle=False, drop_last=False, n_cores=4)
    rcd = defaultdict(list)
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            data = {k: v.to(opts.device).float() for k, v in data.items()}
            X, X_normal = data["src_pcd"].clone(), data["src_pcd_normal"].clone()
            Y, Y_normal = data["model_pcd"].clone(), data["model_pcd_normal"].clone()
            B = len(X)
            H_t = torch.eye(4)[None].expand(B, -1, -1).to(opts.device)  # [B, 4, 4]

            X_list = []
            for t in range(opts.n_diff_steps, 1, -1):  # [T, T-1, ..., 1]
                X_t = (H_t[:, :3, :3] @ X.transpose(2, 1) + H_t[:, :3, [3]]).transpose(2, 1)  # [B, N, 3]
                X_list.append(X_t[0].cpu().numpy())
                X_normal_t = (H_t[:, :3, :3] @ X_normal.transpose(2, 1)).transpose(2, 1)  # [B, N, 3]
                Rs_pred, ts_pred = surrogate_model.forward({
                    "src_pcd": X_t,
                    "src_pcd_normal": X_normal_t,
                    "model_pcd": Y,
                    "model_pcd_normal": Y_normal
                })
                _delta_H_t = torch.cat([Rs_pred, ts_pred.unsqueeze(-1)], dim=2)  # [B, 3, 4]
                delta_H_t = torch.eye(4)[None].expand(B, -1, -1).to(opts.device)  # [B, 4, 4]
                delta_H_t[:, :3, :] = _delta_H_t
                H_0 = delta_H_t @ H_t

                gamma0 = opts.vs.gamma0[t]
                gamma1 = opts.vs.gamma1[t]
                H_t = se3.exp(gamma0 * se3.log(H_0) + gamma1 * se3.log(H_t))

                ### noise
                if opts.is_add_noise:
                    alpha_bar = opts.vs.alpha_bars[t]
                    alpha_bar_ = opts.vs.alpha_bars[t-1]
                    beta = opts.vs.betas[t]
                    cc = ((1 - alpha_bar_) / (1.- alpha_bar)) * beta
                    scale = torch.cat([torch.ones(3) * opts.sigma_r, torch.ones(3) * opts.sigma_t])[None].to(opts.device)  # [1, 6]
                    noise = torch.sqrt(cc) * scale * torch.randn(B, 6).to(opts.device)  # [B, 6]
                    H_noise = se3.exp(noise)
                    H_t = H_noise @ H_t  # [B, 4, 4]

            Rs_pred = H_0[:, :3, :3]
            ts_pred = H_0[:, :3, 3]
            Rs_pred = torch.inverse(Rs_pred)
            ts_pred =  (- Rs_pred @ ts_pred[:, :, None])[:, :, 0]

            rcd["Rs_pred"].extend(list(Rs_pred.cpu().numpy()))
            rcd["ts_pred"].extend(list(ts_pred.cpu().numpy()))
            rcd["Rs_gt"].extend(list(data["R_gt_ms"].cpu().numpy()))
            rcd["ts_gt"].extend(list(data["t_gt_ms"].cpu().numpy()))

        if opts.is_save_res:
            save_data(opts.save_path, rcd)

    print(opts.save_path)

def cal_score():
    res_paths = [
        "./results/DiffusionReg-DiffusionDCP-tudl-diffusion_200_0.00010_0.05_0.05_0.03-nvids3_cosine/eval_results/model_epoch19_T5_cosine_tudl_000001_noiseTrue_v1.pth",
        "./results/DiffusionReg-DiffusionDCP-tudl-diffusion_200_0.00010_0.05_0.05_0.03-nvids3_cosine/eval_results/model_epoch19_T5_cosine_tudl_000002_noiseTrue_v1.pth",
        "./results/DiffusionReg-DiffusionDCP-tudl-diffusion_200_0.00010_0.05_0.05_0.03-nvids3_cosine/eval_results/model_epoch19_T5_cosine_tudl_000003_noiseTrue_v1.pth",
    ]
    score = defaultdict(list)
    for res_path in res_paths:
        print(res_path)
        res = load_data(res_path)
        scale = 256
        score["Rs_gt"].extend(res["Rs_gt"])
        score["Rs_pred"].extend(res["Rs_pred"])
        score["ts_gt"].extend([x * scale / 10. for x in res["ts_gt"]])
        score["ts_pred"].extend([x * scale / 10. for x in res["ts_pred"]])
        auc_R, auc_t = mAP(score["Rs_pred"], score["ts_pred"], score["Rs_gt"], score["ts_gt"])
    print("mAP_R (5/10/20 degree): %.3f, %.3f, %.3f | mAP_t (1/2/5 cm): %.3f, %.3f, %.3f |" % (*auc_R[:3],  *auc_t[:3]))
    print(np.mean(score["times"]))

if __name__ == '__main__':
    opts.db_nm = "tudl"
    for cls_nm in ["000001", "000002", "000003"]:
        opts.vid_infos = [cls_nm]
        main(opts)

    # cal_score()
