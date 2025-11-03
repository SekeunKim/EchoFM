## EchoFM - A Video Vision Foundation Model for Echocardiogram

Official repo for [EchoFM: Foundation Model for Generalizable  Echocardiogram Analysis]

This model and associated code are released under the CC-BY-NC-ND 4.0 license and may only be used for non-commercial, academic research purposes with proper attribution. Any commercial use, sale, or other monetization of the EchoFM model and its derivatives, which include models trained on outputs from the EchoFM model or datasets created from the EchoFM model, is prohibited and requires prior approval. 

<img src="./figure/fig1.png" width="800px"></img>

This work was supported by the National Research Foundation of Korea(NRF) grant funded by the Korea government(MSIT) (RS-2024-00348696)

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
[EchoFM Weights](https://drive.google.com/drive/folders/1Gn43_qMwk-wzZIxZdxXLyk2mXDv5Jsxt?usp=share_link)

## 3. Citation
If you find this repository useful, please consider citing this paper: [will be released soon]
```
@article{kim2024echofm,
  title={EchoFM: Foundation Model for Generalizable Echocardiogram Analysis},
  author={Kim, Sekeun and Jin, Pengfei and Song, Sifan and Chen, Cheng and Li, Yiwei and Ren, Hui and Li, Xiang and Liu, Tianming and Li, Quanzheng},
  journal={arXiv preprint arXiv:2410.23413},
  year={2024}
}
```
