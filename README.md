# SE(3) Diffusion Model-based Point Cloud Registration for Robust 6D Object Pose Estimation (NeurIPS2023)

PyTorch implementation of the paper ["SE(3) Diffusion Model-based Point Cloud Registration for Robust 6D Object Pose Estimation"](https://openreview.net/pdf?id=Znpz1sv4IP).

Haobo Jiang, Mathieu Salzmann, Zheng Dang, Jin Xie, and Jian Yang.

Here is the [supplementary material](https://openreview.net/attachment?id=Znpz1sv4IP&name=supplementary_material).


## Introduction

In this paper, we introduce an SE(3) diffusion model-based point cloud registration framework for 6D object pose estimation in real-world scenarios. Our approach formulates the 3D registration task as a denoising diffusion process, which progressively refines the pose of the source point cloud to obtain a precise alignment with the model point cloud.
Training our framework involves two operations: An SE(3) diffusion process and an SE(3) reverse process. The SE(3) diffusion process gradually perturbs the optimal rigid transformation of a pair of point clouds by continuously injecting noise (perturbation transformation). 
By contrast, the SE(3) reverse process focuses on learning a denoising network that refines the noisy transformation step-by-step, bringing it closer to the optimal transformation for accurate pose estimation. Unlike standard diffusion models used in linear Euclidean spaces, our diffusion model operates on the SE(3) manifold. 
This requires exploiting the linear Lie algebra se(3) associated with SE(3) to constrain the transformation transitions during the diffusion and reverse processes. Additionally, to effectively train our denoising network, we derive a registration-specific variational lower bound as the optimization objective for model learning. 
Furthermore, we show that our denoising network can be constructed with a surrogate registration model, making our approach applicable to different deep registration networks. Extensive experiments demonstrate that our diffusion registration framework presents outstanding pose estimation performance on the real-world TUD-L, LINEMOD, and Occluded-LINEMOD datasets.


## Dataset Preprocessing

### TUD-L

The raw data of TUD-L can be downloaded from BOP datasets: [training data](https://huggingface.co/datasets/bop-benchmark/datasets/resolve/main/tudl/tudl_train_real.zip), [testing data](https://huggingface.co/datasets/bop-benchmark/datasets/resolve/main/tudl/tudl_test_bop19.zip) and [object models](https://huggingface.co/datasets/bop-benchmark/datasets/resolve/main/tudl/tudl_models.zip).
Also, please download pre-processed files: [train_info.pth](https://drive.google.com/file/d/1p07nibykEeVPrXzQf69pWPIAE8GnjDXC/view?usp=sharing), [test_info.pth](https://drive.google.com/file/d/16CeFZ9hfUnh1eoisx7cPzWftEfZCfO9w/view?usp=sharing), and [model_info.pth](https://drive.google.com/file/d/1yFu56Wmr-DFiWmWfYT66SaThmRnaFcAi/view?usp=sharing) 
Please put them into the directory: `./datasets/tudl/` as below:
```
.                          
├── train                 
│   ├── 000001       
│   ├── 000002    
│   └── 000003                
├── test                   
│   ├── 000001   
│   ├── 000002
│   └── 000003
├── models 
│   ├── models_info.json   
│   ├── obj_000001.ply
│   └── obj_000002.ply  
│   └── obj_000003.ply 
├── train_info.pth      
├── test_info.pth
├── models_info.pth                                          
```

## Pretrained Model

We provide the pre-trained model of Diff-DCP on TUD-L dataset in `./results/DiffusionReg-DiffusionDCP-tudl-diffusion_200_0.00010_0.05_0.05_0.03-nvids3_cosine/model_epoch19.pth`.

## Instructions to training and testing

The training and testing can be done by running
```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py --net_type DiffusionDCP --db_nm tudl

CUDA_VISIBLE_DEVICES=0 python3 test.py
```

## Citation

If you find this project useful, please cite:

```bash
@inproceedings{jiang2023se,
  title={SE (3) Diffusion Model-based Point Cloud Registration for Robust 6D Object Pose Estimation},
  author={Jiang, Haobo and Salzmann, Mathieu and Dang, Zheng and Xie, Jin and Yang, Jian},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```

## Acknowledgments
We thank the authors of 
- [DCP](https://github.com/WangYueFt/dcp)
- [RPMNet](https://github.com/yewzijian/RPMNet)

for open sourcing their methods.
