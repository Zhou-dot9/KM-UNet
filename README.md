# KM-UNet
KM-UNet is a deep learning model based on the KAN-Mamba hybrid network for cloud mask nowcasting.
<img width="1940" height="1107" alt="model-第 15 页 drawio" src="https://github.com/user-attachments/assets/b38d6d2f-29c2-4217-b644-702c57ee8d88" />

1. To obtain the Shanghai dataset, please visit [DiffCast](https://github.com/DeminYu98/DiffCast).

2. To create and activate the environment, run
```bash
conda create -n yourname python=3.8
conda activate yourname
pip install -r requirements.txt
```
3. To train models, cd to the project's root directory and run
```bash
python train_shanghai.py
```
This project is based on [Kolmogorov-Arnold Networks](https://github.com/KindXiaoming/pykan), [EfficientViM](https://github.com/mlvlab/EfficientViM), [DiffCast](https://github.com/DeminYu98/DiffCast), thanks for their excellent works.
