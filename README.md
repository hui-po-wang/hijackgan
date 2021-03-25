# Hijack-GAN: Unintended-Use of Pretrained, Black-Box GANs (Still under Construction ...)
In this work, we propose a framework HijackGAN, which enables non-linear latent space traversal and gain high-level controls, e.g., attributes, head poses, and landmarks, over unconditional image generation GANs in a fully black-box setting. It opens up the possibility of reusing GANs while raising concerns about unintended usage.

[[Paper (CVPR 2021)]](https://arxiv.org/abs/2011.14107)[[Project Page]](https://a514514772.github.io/hijackgan/)
## Prerequisites
### Install required packages
### Download pretrained GANs
## Quick Start
Specify number of images to edit, a model to generate images, some parameters for editting.
```
LATENT_CODE_NUM=1
python edit.py \
    -m pggan_celebahq \
    -b boundaries/ \
    -n "$LATENT_CODE_NUM" \
    -o results/stylegan_celebahq_smile_editing \
    --step_size 0.2 \
    --steps 40 \
    --attr_index 0 \
    --task attribute \
    --method ours
```
## Usage
**Important:** For different given images (initial points), different step size and steps may be considered. In the following examples, we provide the parameters used in our paper. One could adjust them for better performance.

### Specify Number of Samples
```
LATENT_CODE_NUM=1
```
### Unconditional Modification
```
python edit.py \
    -m pggan_celebahq \
    -b boundaries/ \
    -n "$LATENT_CODE_NUM" \
    -o results/stylegan_celebahq_smile_editing \
    --step_size 0.2 \
    --steps 40 \
    --attr_index 0\
    --task attribute
```

### Conditional Modification
```
python edit.py \
    -m pggan_celebahq \
    -b boundaries/ \
    -n "$LATENT_CODE_NUM" \
    -o results/stylegan_celebahq_smile_editing \
    --step_size 0.2 \
    --steps 40 \
    --attr_index 0\
    --condition\
    -i codes/pggan_cond/age.npy
    --task attribute
```

### Head pose
#### Pitch
```
python edit.py \
    -m stylegan_celebahq \
    -b boundaries/ \
    -n "$LATENT_CODE_NUM" \
    -o results/ \
    --task head_pose \
    --method ours \
    --step_size 0.01 \
    --steps 2000 \
    --attr_index 1\
    --condition\
    --direction -1 \
    --demo
```
#### Yaw
```
python edit.py \
    -m stylegan_celebahq \
    -b boundaries/ \
    -n "$LATENT_CODE_NUM" \
    -o results/ \
    --task head_pose \
    --method ours \
    --step_size 0.1 \
    --steps 200 \
    --attr_index 0\
    --condition\
    --direction 1\
    --demo
```
### Landmarks
Parameters for reference: (attr_index, step_size, steps) (4: 0.005 400) (5: 0.01 100), (6: 0.1 200), (8 0.1 200)
```
CUDA_VISIBLE_DEVICES=0 python edit.py \
    -m stylegan_celebahq \
    -b boundaries/ \
    -n "$LATENT_CODE_NUM" \
    -o results/ \
    --task landmark \
    --method ours \
    --step_size 0.1 \
    --steps 200 \
    --attr_index 6\
    --condition\
    --direction 1 \
    --demo
```
### Generate Balanced Data
## Evaluations

## Acknowledgment
This code is built upon [InterfaceGAN](https://github.com/genforce/interfacegan)
