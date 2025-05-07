# Virtual Try-On with Latent Diffusion Models

This project explores two conditioning strategies for virtual try-on using latent diffusion models (LDMs), based on Stable Diffusion v1.5:

- **Direct Concatenation**: A simplified architecture with all inputs merged as latent channels.
- **Dual ControlNet**: Uses two separate ControlNets for pose and segmentation conditioning.

**Dataset**: [VITON-HD](https://www.kaggle.com/datasets/marquis03/high-resolution-viton-zalando-dataset)  
**Metrics**: FID, LPIPS, SSIM

## Setup

```bash
conda create -n viton python=3.8
conda activate viton
git clone https://github.com/Yanko96/cs682-project.git
cd cs682-project
pip install -r requirements.txt
```

## Data Preprocessing

### Download VITON-HD dataset from Kaggle:
https://www.kaggle.com/datasets/marquis03/high-resolution-viton-zalando-dataset

### Unzip the dataset:
```bash
unzip VITON-HD.zip -d VITON-HD/
```

### Preprocess into .pt format:

```bash
python preprocess_viton_hd.py --data_root VITON-HD/ --output_dir preprocessed_data_v2
```

## Train
### Direct Concatenation
```bash
python train_direct_concat.py ...
```

### Dual ControlNet
```bash
python train_dual_controlnet.py ...
```

## Test
### Direct Concatenation
```bash
python eval_direct_concat.py ...
```

### Dual ControlNet
```bash
python eval_dual_controlnet.py ...
```