## Usage
Note that for different given images (initial points), different step size and steps may be considered. In the following examples, we provide the parameters used in our paper. One could adjust them for better performance.

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
# Parameters for reference (attr_index, step_size, steps) (4: 0.005 400) (5: 0.01 100), (6: 0.1 200), (8 0.1 200)
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

## Acknowledgment
This code is built upon InterfaceGAN