## EchoFM - A Video Vision Foundation Model for Echocardiogram

Official repo for [EchoFM: Foundation Model for Generalizable  Echocardiogram Analysis]

This model and associated code are released under the CC-BY-NC-ND 4.0 license and may only be used for non-commercial, academic research purposes with proper attribution. Any commercial use, sale, or other monetization of the EchoFM model and its derivatives, which include models trained on outputs from the EchoFM model or datasets created from the EchoFM model, is prohibited and requires prior approval. 

<img src="./figure/fig1.png" width="800px"></img>

## Key features

- EchoFM is pre-trained on 290K Echocardiography clips with self-supervised learning
- EchoFM has been validated in multiple downstream tasks including segmentatino, classification, disease detection tasks.
- EchoFM can be efficiently adapted to customised tasks.

<img src="./figure/fig2.png" width="800px"></img>

## 1. Environment Setup

```bash
git clone https://github.com/SekeunKim/EchoFM.git
cd EchoFM
./environment_setup.sh EchoFM
```

## 2. Download model
Download the EchoFM weights from the following link:  
Pretrained weights are hosted on Hugging Face: [sekeun/EchoFM](https://huggingface.co/sekeun/EchoFM)

## 3. Self-supervised pretraining

EchoFM pretrains a ViT-L video MAE with (i) **spatio-temporal consistent masking** — one spatial mask shared across all frames — and (ii) a **periodic-driven contrastive (triplet) loss** over per-frame CLS embeddings from a shared-weight ViT projector, on top of the masked reconstruction loss.

```bash
# folder of video files (.mp4/.avi/...)
torchrun --nproc_per_node=8 --standalone main_pretrain.py \
    --data_source mp4 --data_path /path/to/videos \
    --model mae_vit_large_patch16 \
    --num_frames 32 --t_patch_size 4 --mask_ratio 0.75 \
    --batch_size 8 --epochs 100 --warmup_epochs 10 --blr 1e-3 \
    --output_dir ./output_dir

# folder of cached clips stored as .npy arrays of shape (T, H, W, 3) uint8
torchrun --nproc_per_node=8 --standalone main_pretrain.py \
    --data_source npy --data_path /path/to/clips [same options]
```

Total loss = masked-patch MSE + triplet loss; both terms are logged separately (`recon`, `triplet`) to stdout and tensorboard.

Unit/smoke tests: `python tests/test_echofm.py [--cuda]`.

A SLURM/apptainer launch script is provided in `cluster/pretrain_apptainer.job`.

## 4. Periodicity verification

The pretrained embeddings are phase-aware: frames one cardiac cycle apart (same phase)
are closer in embedding space than frames half a cycle apart (opposite phase).

<img src="./figure/ed_es_periodicity.png" width="800px"></img>

`notebooks/echofm_usage.ipynb` shows how to extract features for downstream tasks,
run masked reconstruction, and reproduce this periodicity check.

## 5. Citation
If you find this repository useful, please consider citing this paper: [will be released soon]
```
@article{kim2024echofm,
  title={EchoFM: Foundation Model for Generalizable Echocardiogram Analysis},
  author={Kim, Sekeun and Jin, Pengfei and Song, Sifan and Chen, Cheng and Li, Yiwei and Ren, Hui and Li, Xiang and Liu, Tianming and Li, Quanzheng},
  journal={arXiv preprint arXiv:2410.23413},
  year={2024}
}
```
