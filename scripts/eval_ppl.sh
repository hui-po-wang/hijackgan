LATENT_CODE_NUM=1000
CUDA_VISIBLE_DEVICES=7 python ../evaluate_metric.py \
    --metric ppl \
    -m pggan_celebahq \
    -b ../boundaries/ \
    -n "$LATENT_CODE_NUM" \
    -o ./predictions \
    -a 0 \
    --method static \
    --step_size 0.01\
    --steps 600 \
    --condition
