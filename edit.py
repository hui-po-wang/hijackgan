# python3.7
"""Edits latent codes with respect to given boundary.

Basically, this file takes latent codes and a semantic boundary as inputs, and
then shows how the image synthesis will change if the latent codes is moved
towards the given boundary.

NOTE: If you want to use W or W+ space of StyleGAN, please do not randomly
sample the latent code, since neither W nor W+ space is subject to Gaussian
distribution. Instead, please use `generate_data.py` to get the latent vectors
from W or W+ space first, and then use `--input_latent_codes_path` option to
pass in the latent vectors.
"""

import os.path
import argparse
import cv2
import numpy as np
from tqdm import tqdm

from models.model_settings import MODEL_POOL
from models.pggan_generator import PGGANGenerator
from models.stylegan_generator import StyleGANGenerator
from utils.logger import setup_logger
from utils.manipulator import linear_interpolate
from my_interpolation import my_linear_interpolate
from collections import defaultdict
#from search_interpolation import my_linear_interpolate

def process_bound_path(boundary_path, attr_index, gan_type, condition):
    bound_choices = {}
    if gan_type == 'pggan' and not condition:
        bound_choices[0] = 'pggan_celebahq_eyeglasses_boundary.npy'
        bound_choices[1] = 'pggan_celebahq_gender_boundary.npy'
        bound_choices[2] = 'pggan_celebahq_smile_boundary.npy'
        bound_choices[3] = 'pggan_celebahq_age_boundary.npy'
        bound_choices[5] = 'pggan_celebahq_bald.npy'
        bound_choices[7] = 'pggan_celebahq_narrow_eyes.npy'
        bound_choices[10] = 'pggan_celebahq_blonde_hair.npy'
        bound_choices[13] = 'pggan_celebahq_pale_skin.npy'
    elif gan_type == 'stylegan' and not condition:
        bound_choices[0] = 'stylegan_celebahq_eyeglasses_boundary.npy'
        bound_choices[1] = 'stylegan_celebahq_gender_boundary.npy'
        bound_choices[2] = 'stylegan_celebahq_smile_boundary.npy'
        bound_choices[3] = 'stylegan_celebahq_age_boundary.npy'
        bound_choices[5] = 'stylegan_celebahq_bald.npy'
        bound_choices[7] = 'stylegan_celebahq_narrow_eyes.npy'
        bound_choices[10] = 'stylegan_celebahq_blonde_hair.npy'
        bound_choices[13] = 'stylegan_celebahq_pale_skin.npy'
    elif gan_type == 'pggan' and condition:
        bound_choices = ['pggan_celebahq_eyeglasses_c_all_boundary.npy', 'pggan_celebahq_gender_c_all_boundary.npy',
            'pggan_celebahq_smile_c_all_boundary.npy', 'pggan_celebahq_age_c_all_boundary.npy']
    elif gan_type == 'stylegan' and condition:
        bound_choices = ['stylegan_celebahq_eyeglasses_c_all_boundary.npy', 'stylegan_celebahq_gender_c_all_boundary.npy',
            'stylegan_celebahq_smile_c_all_boundary.npy', 'stylegan_celebahq_age_c_all_boundary.npy']
    else:
        raise ValueError('process_bound_path: unknown gan type.')

    print(boundary_path, bound_choices[attr_index], attr_index)
    return os.path.join(boundary_path, bound_choices[attr_index])

def demo_code(gan_type, args):
    out_path = ""
    if gan_type == 'stylegan':
        if args.task == 'attribute':
            if args.condition:
                if args.attr_index == 0:
                    out_path = './codes/stylegan_cond/eyeglasses.npy'
                elif args.attr_index == 1:
                    pass
                elif args.attr_index == 2:
                    pass
                elif args.attr_index == 3:
                    pass
            else:
                if args.attr_index == 5:
                    out_path = './codes/stylegan_rare/bald.npy'
                elif args.attr_index == 7:
                    out_path = './codes/stylegan_rare/narrow_eyes.npy'
                elif args.attr_index == 10:
                    out_path = './codes/stylegan_rare/blond_hair.npy'
                elif args.attr_index == 13:
                    out_path = './codes/stylegan_rare/pale_skin.npy'
                elif args.attr_index in list(range(4)):
                    out_path = './codes/stylegan_uncond/seed2.npy'
        elif args.task == 'landmark':
            if args.attr_index == 4:
                out_path = './codes/landmark/nose_leftright.npy'
            elif args.attr_index == 5:
                out_path = './codes/supp/stylegan_landmark5.npy'
            elif args.attr_index == 6 or args.attr_index == 8:
                out_path = './codes/supp/stylegan_landmark6.npy'
        elif args.task == 'pose':
            if args.attr_index == 0:
                out_path = './codes/supp/stylegan_nose.npy'
            elif args.attr_index == 1:
                out_path = './codes/pose/woman3.npy'
    elif gan_type == 'pggan':
        if args.task == 'attribute':
            if args.condition:
                if args.attr_index == 0:
                    out_path = './codes/pggan_cond/eyeglasses.npy'
                elif args.attr_index == 1:
                    out_path = './codes/pggan_cond/gender.npy'
                elif args.attr_index == 2:
                    out_path = './codes/pggan_cond/smile.npy'
                elif args.attr_index == 3:
                    out_path = './codes/pggan_cond/age.npy'
            else:
                out_path = './codes/pggan_uncond/seed1.npy'
    
    args.input_latent_codes_path = out_path

def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser(
      description='Edit image synthesis with given semantic boundary.')
  parser.add_argument('-m', '--model_name', type=str, required=True,
                      choices=list(MODEL_POOL),
                      help='Name of the model for generation. (required)')
  parser.add_argument('-o', '--output_dir', type=str, required=True,
                      help='Directory to save the output results. (required)')
  parser.add_argument('-b', '--boundary_path', type=str, required=True,
                                            help='Path to the semantic boundary. (required)')
  parser.add_argument('--attr_index', type=int, required=True)
  parser.add_argument('-i', '--input_latent_codes_path', type=str, default='',
                      help='If specified, will load latent codes from given '
                           'path instead of randomly sampling. (optional)')
  parser.add_argument('-n', '--num', type=int, default=1,
                      help='Number of images for editing. This field will be '
                           'ignored if `input_latent_codes_path` is specified. '
                           '(default: 1)')
  parser.add_argument('-s', '--latent_space_type', type=str, default='z',
                      choices=['z', 'Z', 'w', 'W', 'wp', 'wP', 'Wp', 'WP'],
                      help='Latent space used in Style GAN. (default: `Z`)')
  parser.add_argument('--start_distance', type=float, default=-3.0,
                      help='Start point for manipulation in latent space. '
                           '(default: -3.0)')
  parser.add_argument('--end_distance', type=float, default=3.0,
                      help='End point for manipulation in latent space. '
                           '(default: 3.0)')
  parser.add_argument('--steps', type=int, default=10,
                      help='Number of steps for image editing. (default: 10)')
  parser.add_argument('--step_size', type=float, default=0.6,
                      help='Number of steps for image editing. (default: 10)')
  parser.add_argument('--task', type=str, default='attribute',
                      choices=['attribute', 'landmark', 'head_pose'],
                      help='Task to execute. (default: attribute)')
  parser.add_argument('--method', type=str, required=True,
                                            choices=['interfacegan', 'linear', 'ours'])
  parser.add_argument('--condition', action='store_true')
  parser.add_argument('--direction', type=float, default=None, choices=[-1, 1],
                      help='Control if attribute of interest increases or decreases.'
                           'It only works for headpose and landmark.'
                           'The attribute tasks always flip the target automatically.')

  return parser.parse_args()


# +
def main():
    """Main function."""
    args = parse_args()
    
    # append task to the output dir path
    args.output_dir = os.path.join(args.output_dir, args.task)
    # create a directory if the output path does not exist
    #if not os.path.exists(args.output_dir):
    #    os.mkdir(args.output_dir)
        
    logger = setup_logger(args.output_dir, logger_name='generate_data')

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

    logger.info(f'Preparing boundary.')
    args.boundary_path = process_bound_path(gan_type, args)
    if not os.path.isfile(args.boundary_path):
        raise ValueError(f'Boundary `{args.boundary_path}` does not exist!')
    boundary = np.load(args.boundary_path)
    np.save(os.path.join(args.output_dir, 'boundary.npy'), boundary)

    logger.info(f'Preparing latent codes.')
    if args.demo:
        demo_code(gan_type, args)
    
    if os.path.isfile(args.input_latent_codes_path):
        logger.info(f'  Load latent codes from `{args.input_latent_codes_path}`.')
        latent_codes = np.load(args.input_latent_codes_path)
        print(latent_codes.shape)
        if len(latent_codes) > 1:
            latent_codes = np.expand_dims(latent_codes[0], axis=0)
        latent_codes = model.preprocess(latent_codes, **kwargs)
    else:
        logger.info(f'  Sample latent codes randomly.')
        latent_codes = model.easy_sample(args.num, **kwargs)
    np.save(os.path.join(args.output_dir, 'latent_codes.npy'), latent_codes)
    total_num = latent_codes.shape[0]

    logger.info(f'Editing {total_num} samples.')
    for sample_id in tqdm(range(total_num), leave=False):
        attr_index = args.attr_index
        if args.task == 'attribute':
            if args.method == 'interfacegan':
                # baseline modification from initial point
                interpolations = my_linear_interpolate(latent_codes[sample_id:sample_id + 1],
                                                    attr_index,
                                                    boundary,
                                                    'linear',
                                                    steps=args.steps,
                                                    gan_type=gan_type,
                                                    step_size=args.step_size)
                interpolation_id = 0
                for interpolations_batch in model.get_batch_inputs(interpolations):
                    if gan_type == 'pggan':
                        outputs = model.easy_synthesize(interpolations_batch)
                    elif gan_type == 'stylegan':
                        outputs = model.easy_synthesize(interpolations_batch, **kwargs)
                    for image in outputs['image']:
                        save_path = os.path.join(args.output_dir,
                                            f'{sample_id:03d}_{interpolation_id:03d}.jpg')
                        cv2.imwrite(save_path, image[:, :, ::-1])
                        interpolation_id += 1

            elif args.method == 'linear':
                # linear baseline attribute modification
                starting_latent_code = latent_codes[sample_id:sample_id + 1].reshape(1, -1)
                interpolations = my_linear_interpolate(starting_latent_code,
                                                    attr_index,
                                                    boundary,
                                                    'static_linear',
                                                    steps=args.steps,
                                                    condition=args.condition,
                                                    gan_type=gan_type,
                                                    step_size=args.step_size)
                interpolation_id = 0
                for interpolations_batch in model.get_batch_inputs(interpolations):
                    if gan_type == 'pggan':
                        outputs = model.easy_synthesize(interpolations_batch)
                    elif gan_type == 'stylegan':
                        outputs = model.easy_synthesize(interpolations_batch, **kwargs)
                    for image in outputs['image']:
                        save_path = os.path.join(args.output_dir,
                                             f'{sample_id:03d}_{interpolation_id:03d}.jpg')
                        cv2.imwrite(save_path, image[:, :, ::-1])
                        interpolation_id += 1

            elif args.method == 'ours':
                # attribute modification
                starting_latent_code = latent_codes[sample_id:sample_id + 1].reshape(1, -1)
                interpolations = my_linear_interpolate(starting_latent_code,
                                                        attr_index,
                                                        boundary,
                                                        'piecewise_linear',
                                                        steps=args.steps,
                                                        condition=args.condition,
                                                        gan_type=gan_type,
                                                        step_size=args.step_size)
                interpolation_id = 0
                for interpolations_batch in model.get_batch_inputs(interpolations):
                    if gan_type == 'pggan':
                        outputs = model.easy_synthesize(interpolations_batch)
                    elif gan_type == 'stylegan':
                        outputs = model.easy_synthesize(interpolations_batch, **kwargs)
                    for image in outputs['image']:
                        save_path = os.path.join(args.output_dir,
                                             f'{sample_id:03d}_{interpolation_id:03d}.jpg')
                        cv2.imwrite(save_path, image[:, :, ::-1])
                        interpolation_id += 1
                
        elif args.task == 'head_pose':
            # pose modification
            starting_latent_code = latent_codes[sample_id:sample_id + 1].reshape(1, -1)
            interpolations = my_linear_interpolate(starting_latent_code,
                                                    attr_index,
                                                    boundary,
                                                    'pose_edit',
                                                    steps=args.steps,
                                                    condition=args.condition,
                                                    gan_type=gan_type,
                                                    step_size=args.step_size,
                                                    direction=args.direction)
            interpolation_id = 0
            for interpolations_batch in model.get_batch_inputs(interpolations):
                if gan_type == 'pggan':
                    outputs = model.easy_synthesize(interpolations_batch)
                elif gan_type == 'stylegan':
                    outputs = model.easy_synthesize(interpolations_batch, **kwargs)
                for image in outputs['image']:
                    save_path = os.path.join(args.output_dir,
                                         f'{sample_id:03d}_{interpolation_id:03d}.jpg')
                    cv2.imwrite(save_path, image[:, :, ::-1])
                    interpolation_id += 1
                
        elif args.task == 'landmark':
            # landmark modification
            starting_latent_code = latent_codes[sample_id:sample_id + 1].reshape(1, -1)
            interpolations = my_linear_interpolate(starting_latent_code,
                                                    attr_index,
                                                    boundary,
                                                    'piecewise_linear',
                                                    steps=args.steps,
                                                    is_landmark=True,
                                                    condition=args.condition,
                                                    step_size=args.step_size,
                                                    direction=args.direction)
            interpolation_id = 0
            for interpolations_batch in model.get_batch_inputs(interpolations):
                if gan_type == 'pggan':
                    outputs = model.easy_synthesize(interpolations_batch)
                elif gan_type == 'stylegan':
                    outputs = model.easy_synthesize(interpolations_batch, **kwargs)
                for image in outputs['image']:
                    save_path = os.path.join(args.output_dir,
                                         f'{sample_id:03d}_{interpolation_id:03d}.jpg')
                    cv2.imwrite(save_path, image[:, :, ::-1])
                    interpolation_id += 1


if __name__ == '__main__':
  main()
