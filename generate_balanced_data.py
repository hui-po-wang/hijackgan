# python3.7
"""Generates a collection of images with specified model.

Commonly, this file is used for data preparation. More specifically, before
exploring the hidden semantics from the latent space, user need to prepare a
collection of images. These images can be used for further attribute prediction.
In this way, it is able to build a relationship between input latent codes and
the corresponding attribute scores.
"""

import os.path
import argparse
import pickle
import torch
import torch.nn as nn

from collections import defaultdict
import cv2
import numpy as np
from tqdm import tqdm

from models.classifiers import EvalCompoundResNet
from models.model_settings import MODEL_POOL
from models.pggan_generator import PGGANGenerator
from models.stylegan_generator import StyleGANGenerator
from utils.logger import setup_logger

def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser(
      description='Generate images with given model.')
  parser.add_argument('-m', '--model_name', type=str, required=True,
                      choices=list(MODEL_POOL),
                      help='Name of the model for generation. (required)')
  parser.add_argument('-o', '--output_dir', type=str, required=True,
                      help='Directory to save the output results. (required)')
  parser.add_argument('-i', '--latent_codes_path', type=str, default='',
                      help='If specified, will load latent codes from given '
                           'path instead of randomly sampling. (optional)')
  parser.add_argument('-n', '--num', type=int, default=1,
                      help='Number of images to generate. This field will be '
                           'ignored if `latent_codes_path` is specified. '
                           '(default: 1)')
  parser.add_argument('-s', '--latent_space_type', type=str, default='z',
                      choices=['z', 'Z', 'w', 'W', 'wp', 'wP', 'Wp', 'WP'],
                      help='Latent space used in Style GAN. (default: `Z`)')
  parser.add_argument('-S', '--generate_style', action='store_true',
                      help='If specified, will generate layer-wise style codes '
                           'in Style GAN. (default: do not generate styles)')
  parser.add_argument('-I', '--generate_image', action='store_false',
                      help='If specified, will skip generating images in '
                           'Style GAN. (default: generate images)')
  parser.add_argument('-K', '--key_file_path', type=str, default='./data/bald.pkl',
                      help='Path to save the file that records the indices of pos and neg examples.')
  parser.add_argument('-SI', '--start_index', type=int, required=True,
                      help='The first index that is used to name the save image.')
  parser.add_argument('--pretrained_clf_path', type=str, 
                      default='./celebA-hq-classifier')
  parser.add_argument('--sample_per_category', type=int, default=50000)
  parser.add_argument('--threshold', type=float, default=0.9,
                     help='Confidence for samples to be accepted.')
  parser.add_argument('--no_generated_imgs', action='store_true',
                     help='If specified, the program will only keep the noise-task_output pairs; thus save the disk space.')

  return parser.parse_args()

def main():
  """Main function."""
  args = parse_args()
  logger = setup_logger(args.output_dir, logger_name='generate_data', immutable=False)

  logger.info(f'Initializing generator.')
  gan_type = MODEL_POOL[args.model_name]['gan_type']
  if gan_type == 'pggan':
    model = PGGANGenerator(args.model_name, logger)
    kwargs = {}
  elif gan_type == 'stylegan':
    model = StyleGANGenerator(args.model_name, logger)
    kwargs = {'latent_space_type': args.latent_space_type}
  else:
    raise NotImplementedError(f'Not implemented GAN type `{gan_type}`!')

  logger.info(f'Preparing latent codes.')
  if os.path.isfile(args.latent_codes_path):
    logger.info(f'  Load latent codes from `{args.latent_codes_path}`.')
    latent_codes = np.load(args.latent_codes_path)
    latent_codes = model.preprocess(latent_codes, **kwargs)
  else:
    logger.info(f'  Sample latent codes randomly.')
    latent_codes = model.easy_sample(args.num, **kwargs)
    # The orginal code of interfaceGAN does not have this line.
    #if gan_type == 'stylegan':
    #    latent_codes = model.preprocess(latent_codes, **kwargs)
  total_num = latent_codes.shape[0]

  logger.info(f'Generating {total_num} samples.')
  results = defaultdict(list)
  pbar = tqdm(total=total_num, leave=False)
    
  """Pretrained attribute classifier
  
  Replace the classifier 'attr_clf' with your own task models.
  """
  attr_num = 16
  attr_clf = EvalCompoundResNet(args.pretrained_clf_path)
  logger.info(f'Classifier loaded.')
  
  attr_clf.cuda()
  attr_clf.eval()
  attr_clf.requires_grad = False
  
  downscale = nn.Upsample(size=224, mode='bilinear')
  if os.path.exists(args.key_file_path):
    raise ValueError(f'{args.key_file_path} has existed.')
  else:
    data_index = [[[],[]] for _ in range(int(attr_num))]
  image_cnt = args.start_index

  for latent_codes_batch in model.get_batch_inputs(latent_codes):
    if gan_type == 'pggan':
      outputs = model.easy_synthesize(latent_codes_batch)
    elif gan_type == 'stylegan':
      outputs = model.easy_synthesize(latent_codes_batch,
                                      **kwargs,
                                      generate_style=args.generate_style,
                                      generate_image=args.generate_image)
    
    with torch.no_grad():
        val = outputs['image']
        # Within a batch, some of images will be saved, while some of them won't.
        kept_indices = []
        img_tensor = torch.tensor(val.transpose(0, 3, 1, 2)).cuda().float()
        img_tensor = downscale(img_tensor)/255.
        # predict = torch.sigmoid(logits)
        preds = attr_clf.predict(img_tensor)
        for iid, image in enumerate(val): 
          pbar.update(1)
          is_save = False
          for ind in range(attr_num):
            if preds[iid][ind] >= args.threshold and len(data_index[ind][0])<args.sample_per_category:
              is_save = True
              data_index[ind][0].append(image_cnt)
            elif (1 - preds[iid][ind]) >= args.threshold and len(data_index[ind][1])<args.sample_per_category:
              is_save = True
              data_index[ind][1].append(image_cnt)

          if is_save:
            kept_indices.append(iid)
            save_path = os.path.join(args.output_dir, f'{image_cnt:08d}.jpg')
            if not args.no_generated_imgs:
                cv2.imwrite(save_path, image[:, :, ::-1])
            results['soft_labels'].append(preds[iid].reshape(1, -1))
            image_cnt += 1

        for key, val in outputs.items():
            if key != 'image' and len(kept_indices) > 0:
                val = val[kept_indices]
                results[key].append(val)
        if 'image' not in outputs:
          pbar.update(latent_codes_batch.shape[0])
        if pbar.n % 1000 == 0 or pbar.n == total_num:
          print('iter: ', pbar.n)
          for ind in range(attr_num):
            print ("attr_index: {}, pos: {}, neg{}".format(ind, len(data_index[ind][0]), len(data_index[ind][1])))
          # save data_index
          with open(args.key_file_path, 'wb') as f:
            pickle.dump(data_index, f)
          logger.debug(f'  Finish {pbar.n:6d} samples.')
          logger.info(f'Saving results.')
          for key, val in results.items():
            save_path = os.path.join(args.output_dir, f'{key}.npy')
            np.save(save_path, np.concatenate(val, axis=0))
  pbar.close()


if __name__ == '__main__':
  main()
