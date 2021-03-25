# +
import argparse
import os
import pickle
import sys
sys.path.append("..")

import numpy as np
import torchvision
import torchvision.transforms as T
import torch.utils.data as torch_data

from tqdm import tqdm
from models.classifiers import EvalCompoundResNet


# -

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-F', '--function', type=str, required=True, choices=['max_index', 'count_data'])
    parser.add_argument('-O', '--output_path', type=str, required=True)
    parser.add_argument('--num_attr', type=str, default=8)
    parser.add_argument('--sample_per_category', type=int, default=1e5)
    parser.add_argument('--weight_path', type=str, default='/home/u5397696/interpolation/celebA-hq-classifier/')
    parser.add_argument('--data_root', type=str, default='/home/u5397696/interpolation/interfacegan/data/tmp')
    
    return parser.parse_args()


def max_index(args):
    if not os.path.exists(args.output_path):
        raise ValueError(f"{args.output_path} doesn't exist.")
        
    with open(args.output_path, 'rb') as f:
        data_index = pickle.load(f)
    print(f'#attributes: {len(data_index)}')
    
    max_val = -1e9
    for i in range(len(data_index)):
        max_p = np.max(data_index[i][0])
        max_n = np.max(data_index[i][1])
        max_val = np.max([max_val, max_p, max_n])
        
        print(i, max_p, max_n)
    print (f'Max index is {max_val}')


def count_data(args):
    #if os.path.exists(args.output_path):
    #    raise ValueError(f"{args.output_path} has existed.")
        
    t = T.Compose([T.Resize(224), T.ToTensor()])
    dset = torchvision.datasets.ImageFolder(args.data_root, transform=t)

    loader= torch_data.DataLoader(dset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    print (f'Start processing {os.path.basename(args.data_root)}.')
    m = EvalCompoundResNet(args.weight_path).cuda()
    
    data_index = [[[],[]] for _ in range(args.num_attr)]
    image_cnt = 0
    for bid, (imgs, _) in enumerate(loader):
        imgs = imgs.cuda()
        preds = m.predict_quantize(imgs)
        for iid, pred in enumerate(preds):
            is_save = False
            for ind in range(args.num_attr):
                if pred[ind] == True and len(data_index[ind][0])<args.sample_per_category:
                    is_save = True
                    data_index[ind][0].append(image_cnt)
                elif pred[ind] == False and len(data_index[ind][1])<args.sample_per_category:
                    is_save = True
                    data_index[ind][1].append(image_cnt)
            if is_save:
                image_cnt += 1
        
        if bid % 10 == 0:
            for i in range(args.num_attr):
                print(i, len(data_index[i][0]), len(data_index[i][1]))
            print(f'Processes {bid}/{len(loader)}.')
            with open(args.output_path, 'wb') as f:
                pickle.dump(data_index, f)


def main():
    args = parse_args()
    
    if args.function == 'max_index':
        max_index(args)
    elif args.function == 'count_data':
        count_data(args)


if __name__ == '__main__':
    main()
