import torch
from utils.attr_dict import AttrDict

opts = AttrDict()
opts.is_debug = False
opts.is_normal = True
opts.batch_size = 32
opts.n_workers = 12
opts.n_epoches = 30
opts.lr = 0.001
opts.n_input_feats = 0
opts.seed = 1
opts.is_completion = True
opts.device = torch.device("cuda")

# model config
opts.emb_nn = ["rpmnet_emb", "dgcnn", "pointnet"][0]
opts.pointer = "transformer"
opts.head = ["svd", "mlp"][0]
opts.emb_dims = 96
opts.n_blocks = 1
opts.n_heads = 4
opts.ff_dims = 256
opts.dropout = 0.0
if opts.emb_nn in ["rpmnet_emb"]:
    opts.features = ['ppf', 'dxyz', 'xyz']
    opts.feat_dim = 96
    opts.num_neighbors = 64
    opts.radius = 0.3