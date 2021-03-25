NUM=500000
#CUDA_VISIBLE_DEVICES=0 python generate_balanced_data.py -m stylegan_celebahq -o data/stylegan_celebahq -K ./data/data_index_1.pkl -n "$NUM" -SI 0
CUDA_VISIBLE_DEVICES=2 python generate_balanced_data.py -m stylegan_celebahq -o /work/u5397696/data/tmp100 -K ./data/data_index_100.pkl -n "$NUM" -SI 0 --no_generated_imgs
#CUDA_VISIBLE_DEVICES=2 python generate_balanced_data.py -m stylegan_celebahq -o /work/u5397696/data/val -K ./data/data_index_val.pkl -n "$NUM" -SI 0 --sample_per_category 10000
#CUDA_VISIBLE_DEVICES=1 python generate_balanced_data.py -m pggan_celebahq -o /work/u5397696/data/tmp0 -K ./data/data_index_0.pkl -n "$NUM" -SI 0 --no_generated_imgs
#CUDA_VISIBLE_DEVICES=0 python generate_balanced_data.py -m pggan_celebahq -o /work/u5397696/data/val -K ./data/data_index_val.pkl -n "$NUM" -SI 0 --sample_per_category 10000





