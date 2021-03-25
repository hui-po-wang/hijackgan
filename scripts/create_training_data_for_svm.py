# +
import os
import pickle
import argparse

import numpy as np


# -

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--code_path', type=str, default='/home/u5397696/interpolation/interfacegan/data/stylegan_celebahq/z.npy')
    parser.add_argument('-i', '--index_path', type=str, default='/home/u5397696/interpolation/interfacegan/data/stylegan_celebahq/data_index.pkl')
    parser.add_argument('--soft_label_path', type=str, default='/home/u5397696/interpolation/interfacegan/data/stylegan_celebahq/soft_labels.npy')
    parser.add_argument('-O', '--output_dir', type=str, required=True)
    parser.add_argument('--attr_index', type=int, required=True)

    return parser.parse_args()


def main():
    args = parse_args()
    
    if not os.path.isfile(args.code_path):
        raise ValueError(f'Latent codes `{args.code_path}` does not exist!')
    if not os.path.isfile(args.index_path):
        raise ValueError(f'Index file `{args.index_path}` does not exist!')
    
    if not os.path.exists(args.output_dir):
        print(f'Create Path {args.output_dir}')
        os.mkdir(args.output_dir)
    
    with open(args.index_path, 'rb') as f:
        data_index = pickle.load(f)
        noise = np.load(args.code_path)
        soft_labels = np.load(args.soft_label_path)
    
    pos_indices = np.array(data_index[args.attr_index][0])
    neg_indices = np.array(data_index[args.attr_index][1])
    
    min_length = np.min([len(pos_indices), len(neg_indices)])
    pos_indices = pos_indices[np.random.choice(len(pos_indices), min_length, replace=False)]
    neg_indices = neg_indices[np.random.choice(len(neg_indices), min_length, replace=False)]
    
    print(len(pos_indices), len(neg_indices))
    combined_indices = np.concatenate([pos_indices, neg_indices], axis=0)
    print(len(combined_indices))
    selected_noise = noise[combined_indices]
    #selected_labels = [1] * len(pos_indices) + [0] * len(neg_indices)
    selected_labels = soft_labels[combined_indices, args.attr_index]
    selected_labels = np.array(selected_labels).reshape(-1, 1)
    print(selected_labels.shape, selected_noise.shape)
    
    np.save(os.path.join(args.output_dir, f'{args.attr_index}_z.npy'), selected_noise)
    np.save(os.path.join(args.output_dir, f'{args.attr_index}_labels.npy'), selected_labels)


if __name__ == "__main__":
    main()
