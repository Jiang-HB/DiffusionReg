import os, numpy as np, open3d as o3d, copy, pdb
from torch.utils.data import Dataset
from utils.commons import load_data, save_data, regularize_pcd, crop_pcd, depth2pcd, cal_normal
from utils.data_classes import PointCloud
from pyquaternion import Quaternion
from tqdm import tqdm
from collections import defaultdict

class BaseTUDLDataset(Dataset):
    def __init__(self, opts, partition, cls_nms):
        self.opts = opts
        self.db_nm = opts.db_nm
        self.is_test = opts.is_test
        self.partition = partition
        self.is_debug = opts.is_debug
        self.cls_nms = cls_nms
        self.n_points = 1024
        self.scale = 256
        self.depth_expand_ratio = 0.8

        self.base_dir = "./datasets/tudl/"
        self.model_path = "./datasets/tudl/models_info.pth"
        self.gen_db_dir = "./datasets/tudl/"
        if partition == "train":
            self.db_dir = os.path.join(self.base_dir, "train")
            self.db_path = "./datasets/tudl/train_info.pth"
        else:
            self.db_dir = os.path.join(self.base_dir, "test")
            self.db_path = "./datasets/tudl/test_info.pth"

        # load info
        self.models_info = self.get_models_info()
        self.annos_info, self.cropped_depths, self.bboxs, self.annos_list = self.get_annos_info()

    def __len__(self):
        return len(self.cropped_depths)

    def getitem(self, idx):
        raise NotImplementedError

    def __getitem__(self, idx):
        return self.getitem(idx)

    def get_models_info(self):
        models_info = load_data(self.model_path)
        for cls_nm in self.cls_nms:
            model_info = models_info[cls_nm]
            model_pcd = PointCloud(np.asarray(o3d.io.read_point_cloud(os.path.join(self.base_dir, model_info["model_pcd_path"])).points).T).points
            model_bbox = model_info["bbox_3d"]

            model_bbox.center /= self.scale
            model_bbox.wlh /= self.scale
            model_pcd /= self.scale

            correct_matrix = np.eye(3) @ np.linalg.inv(model_bbox.rotation_matrix)
            model_bbox.rotate(quaternion=Quaternion(matrix=correct_matrix))
            model_pcd = correct_matrix @ model_pcd

            models_info[cls_nm]["model_pcd"] = model_pcd
            models_info[cls_nm]["model_bbox"] = model_bbox
            models_info[cls_nm]["correct_matrix"] = correct_matrix

        return models_info

    def get_annos_info(self):
        db_info = load_data(self.db_path)
        annos_info = {}
        cropped_depths, bboxs, anno_list = [], [], []
        for cls_nm in self.cls_nms:
            annos = []
            vid_db = db_info[cls_nm]
            for frame_idx, _ in enumerate(vid_db["frame_idxs"]):
                annos.append({
                    "db_dir": self.db_dir,
                    "cls_nm": cls_nm,
                    "camera_info": vid_db["cameras_info"][frame_idx],
                    "rgb_path": vid_db["rgb_paths"][frame_idx],
                    "depth_path": vid_db["depth_paths"][frame_idx],
                    "mask_path": vid_db["mask_paths"][frame_idx],
                    "mask_visib_path": vid_db["mask_visib_paths"][frame_idx],
                    "frame_idx": int(vid_db["frame_idxs"][frame_idx]),
                    "bbox_obj_2d": vid_db["bboxes_obj_2d"][frame_idx],
                    "bbox_visib_2d": vid_db["bboxes_visib_2d"][frame_idx],
                    "bbox_obj_3d": vid_db["bboxes_obj_3d"][frame_idx],
                    "cam_R_m2c": vid_db["cam_Rs_m2c"][frame_idx],
                    "cam_t_m2c": vid_db["cam_ts_m2c"][frame_idx],
                })
            if self.opts.is_debug:
                annos = annos[:100]
            annos_info[cls_nm] = annos
            anno_list.extend(annos)

            gen_depths_bboxes_path = os.path.join(self.gen_db_dir, "%s_%s_%s_depths_bboxes_tiny%d.pth" % (
                                                            self.db_nm, self.partition, cls_nm, int(self.is_debug)))
            if os.path.exists(gen_depths_bboxes_path):
                print("Data exists. Loading...")
                data_info = load_data(gen_depths_bboxes_path)[cls_nm]
                for cropped_depth, bbox in data_info:
                    cropped_depths.append(cropped_depth)
                    bbox.center /= self.scale
                    bbox.wlh /= self.scale
                    bboxs.append(bbox)
            else:
                print("Data not exists. Generating...")
                data_info = defaultdict(list)
                for anno_idx, anno in enumerate(tqdm(annos)):
                    depth = np.asarray(o3d.io.read_image(os.path.join(anno["db_dir"], anno["depth_path"])))  # [H, W]
                    mask = np.asarray(o3d.io.read_image(os.path.join(anno["db_dir"], anno["mask_visib_path"])))  # [H, W]
                    bbox_2d = anno["bbox_obj_2d"]  # [x, y, w, h]
                    min_xy = np.maximum(np.floor(bbox_2d[:2] + bbox_2d[2:] / 2. - bbox_2d[2:] * self.depth_expand_ratio), 0.).astype(np.int32)
                    max_xy = np.minimum(np.ceil(bbox_2d[:2] + bbox_2d[2:] / 2. + bbox_2d[2:] * self.depth_expand_ratio),
                                        np.asarray([depth.shape[1], depth.shape[0]])).astype(np.int32)
                    cropped_depth = depth[min_xy[1]: max_xy[1], min_xy[0]: max_xy[0]]
                    cropped_mask = mask[min_xy[1]: max_xy[1], min_xy[0]: max_xy[0]]
                    data_info[cls_nm].append([[cropped_depth, min_xy, max_xy, *depth.shape, cropped_mask], anno["bbox_obj_3d"]])

                save_data(gen_depths_bboxes_path, data_info)

                for idx, (cropped_depth, bbox) in enumerate(data_info[cls_nm]):
                    cropped_depths.append(cropped_depth)
                    bbox.center /= self.scale
                    bbox.wlh /= self.scale
                    bboxs.append(bbox)

        return annos_info, cropped_depths, bboxs, anno_list

    def depth2pcd_single(self, idx):
        depth, min_xy, max_xy, H, W, mask = self.cropped_depths[idx]
        pcd = depth2pcd(depth, min_xy, max_xy, H, W, self.annos_list[idx]["camera_info"], scale=self.scale).T  # [3, N]
        pcd = pcd[:, (mask == 255).astype(np.bool_).reshape([-1])]
        pcd = crop_pcd(pcd, self.bboxs[idx], offset=0., scale=2)
        return pcd

    def get_trans_gt(self, anno, cls_nm):
        R_gt = anno["cam_R_m2c"]
        t_gt = anno["cam_t_m2c"] / self.scale
        correct_matrix = self.models_info[cls_nm]["correct_matrix"]
        R_gt_correct = R_gt @ np.linalg.inv(correct_matrix)
        return R_gt_correct, t_gt

    def get_src_pcd(self, src_idx):
        src_pcd = self.depth2pcd_single(src_idx)  # [3, N]
        src_gt_bbox = self.bboxs[src_idx]
        if src_pcd.shape[1] <= 50:
            return self.getitem(np.random.randint(0, self.__len__()))
        src_pcd = regularize_pcd(src_pcd, self.n_points // 2, is_test=self.is_test)  # [3, N]
        return src_pcd, src_gt_bbox

    def get_model_pcd(self, cls_nm):
        model_pcd = regularize_pcd(self.models_info[cls_nm]["model_pcd"], self.n_points, is_test=self.is_test)  # [3, M]
        model_gt_bbox = self.models_info[cls_nm]["model_bbox"]
        return model_pcd, model_gt_bbox

class TUDL_DB_Train(BaseTUDLDataset):
    def __init__(self, opts, partition, cls_nms):
        super(TUDL_DB_Train, self).__init__(opts, partition, cls_nms)
        self.cls_nms = cls_nms

    def gen_reg_sample(self, res, src_pcd, model_pcd, src_gt_bbox, model_gt_bbox, sample_idx):

        t_center = - np.mean(src_pcd, axis=1)
        trans_src_pcd = src_pcd + t_center[:, None]
        trans_src_gt_bbox = copy.deepcopy(src_gt_bbox)
        trans_src_gt_bbox = trans_src_gt_bbox.translate(t_center)
        X, X_BBox = trans_src_pcd, trans_src_gt_bbox
        Y = model_pcd

        res["src_pcd"] = X.T.astype(np.float32)
        res["model_pcd"] = Y.T.astype(np.float32)
        if self.opts.is_normal:
            res["src_pcd_normal"] = cal_normal(X.T, radius=self.opts.radius, max_nn=30).astype(np.float32)
            res["model_pcd_normal"] = cal_normal(Y.T, radius=self.opts.radius, max_nn=30).astype(np.float32)

        # rotation label
        R_ms = X_BBox.rotation_matrix
        t_ms = X_BBox.center
        R_sm = np.linalg.inv(R_ms)
        t_sm = (- R_sm @ t_ms[:, None])[:, 0]
        res["transform_gt"] = np.concatenate([R_sm, t_sm[:, None]], axis=1).astype(np.float32)
        return res

    def getitem(self, idx):

        curr_anno = self.annos_list[idx]
        cls_nm = curr_anno["cls_nm"]
        res = {}

        src_pcd, src_gt_bbox = self.get_src_pcd(idx)  # [3, N]
        model_pcd, model_gt_bbox = self.get_model_pcd(cls_nm)  # [3, M]
        res = self.gen_reg_sample(res, src_pcd, model_pcd, src_gt_bbox, model_gt_bbox, idx)
        return res

class TUDL_DB_Test(BaseTUDLDataset):
    def __init__(self, opts, partition, cls_nms):
        super(TUDL_DB_Test, self).__init__(opts, partition, cls_nms)
        self.cls_nms = cls_nms
        assert len(self.cls_nms) == 1

    def getitem(self, idx):

        anno = self.annos_list[idx]
        cls_nm = anno["cls_nm"]

        src_pcd, src_gt_bbox = self.get_src_pcd(idx)  # [3, N]
        model_pcd, model_gt_bbox = self.get_model_pcd(cls_nm)  # [3, M]

        t_ = - np.mean(src_pcd, axis=1)
        trans_src_pcd = src_pcd + t_[:, None]
        trans_src_gt_bbox = copy.deepcopy(src_gt_bbox)
        trans_src_gt_bbox = trans_src_gt_bbox.translate(t_)
        R_ms = trans_src_gt_bbox.rotation_matrix
        t_ms = trans_src_gt_bbox.center
        R_gt, t_gt = self.get_trans_gt(anno, cls_nm)
        R_sm = np.linalg.inv(R_ms)
        t_sm = (- R_sm @ t_ms[:, None])[:, 0]

        res = {
            "src_pcd0": src_pcd.T.astype(np.float32),  # [N, 3]
            "src_pcd": trans_src_pcd.T.astype(np.float32),  # [N, 3]
            "model_pcd": model_pcd.T.astype(np.float32),  # [M, 3]
            "src_pcd_original": trans_src_pcd.T.astype(np.float32),  # [N, 3]
            "model_pcd_original": model_pcd.T.astype(np.float32),  # [N, 3]
            "src_center": trans_src_gt_bbox.center,
            "model_center": model_gt_bbox.center,
            "R_gt_ms": R_ms,
            "t_gt_ms": t_ms,
            "R_gt_sm": R_sm,
            "t_gt_sm": t_sm,
            "R_gt": R_gt,  # [3, 3]
            "t_gt": t_gt,  # [3]
            "transform_gt": np.concatenate([R_sm, t_sm[:, None]], axis=1).astype(np.float32)
        }

        if self.opts.is_normal:
            res["src_pcd_normal"] = cal_normal(trans_src_pcd.T, radius=self.opts.radius, max_nn=30).astype(np.float32)
            res["model_pcd_normal"] = cal_normal(model_pcd.T, radius=self.opts.radius, max_nn=30).astype(np.float32)

        return res

    def __len__(self):
        return len(self.annos_list)