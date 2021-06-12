NUM=500000
CUDA_VISIBLE_DEVICES=0 python generate_balanced_data.py -m stylegan_celebahq -o ./generated_data -K ./generated_data/indices.pkl -n "$NUM" -SI 0 --no_generated_imgs
