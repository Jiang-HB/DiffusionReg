import pickle, open3d as o3d, numpy as np, copy

def load_data(path):
    file = open(path, "rb")
    data = pickle.load(file)
    file.close()
    return data

def save_data(path, data):
    file = open(path, "wb")
    pickle.dump(data, file)
    file.close()

def cal_normal(pcd, radius=0.1, max_nn=30):
    _pcd = o3d.geometry.PointCloud()
    _pcd.points = o3d.utility.Vector3dVector(pcd)
    _pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    # o3d.geometry.estimate_normals(_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    normals = np.asarray(_pcd.normals)
    return normals

def regularize_pcd(pcd, n_points, is_test):
    assert is_test is not None
    if is_test:
        np.random.seed(1)
    idxs = np.random.randint(low=0, high=pcd.shape[1], size=n_points, dtype=np.int64)
    new_pcd = pcd[:, idxs].astype(np.float32)
    return new_pcd

def crop_pcd(pcd, bbox, offset=0, scale=1.0, is_scale_max=False, is_mask=False):
    bbox_tmp = copy.deepcopy(bbox)
    if is_scale_max:
        bbox_tmp.wlh = np.asarray([np.max(bbox_tmp.wlh)] * 3)
    bbox_tmp.wlh = bbox_tmp.wlh * scale
    maxi = np.max(bbox_tmp.corners(), 1) + offset
    mini = np.min(bbox_tmp.corners(), 1) - offset

    x_filt_max = pcd[0, :] < maxi[0]
    x_filt_min = pcd[0, :] > mini[0]
    y_filt_max = pcd[1, :] < maxi[1]
    y_filt_min = pcd[1, :] > mini[1]
    z_filt_max = pcd[2, :] < maxi[2]
    z_filt_min = pcd[2, :] > mini[2]

    close = np.logical_and(x_filt_min, x_filt_max)
    close = np.logical_and(close, y_filt_min)
    close = np.logical_and(close, y_filt_max)
    close = np.logical_and(close, z_filt_min)
    close = np.logical_and(close, z_filt_max)
    if is_mask:
        return pcd[:, close], close
    else:
        return pcd[:, close]

def depth2pcd(depth, min_xy, max_xy, H, W, cam_info, scale=None):
    if scale is None:
        depth = depth / cam_info[[4]]  # [H, W]
    else:
        depth = depth / scale  # [H, W]

    xv = np.arange(W)
    yv = np.arange(H)
    X_np, Y_np = np.meshgrid(xv, yv)  # [H, W]
    x = (X_np[min_xy[1]: max_xy[1], min_xy[0]: max_xy[0]] -  cam_info[2]) * depth / cam_info[0]  # [H, W]
    y = (Y_np[min_xy[1]: max_xy[1], min_xy[0]: max_xy[0]] -  cam_info[3]) * depth / cam_info[1]  # [H, W]
    pcd = np.stack([x, y, depth], axis=2).reshape([-1, 3])  # [H * W, 3]
    return pcd