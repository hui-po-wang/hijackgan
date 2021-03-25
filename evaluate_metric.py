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

# +
import os.path
import argparse
import cv2
import numpy as np
import lpips
import torch
import torch.nn as nn

import torchvision.transforms as T
from tqdm import tqdm
from PIL import Image
# -

from models.classifiers import EvalCompoundResNet
from models.model_settings import MODEL_POOL
from models.pggan_generator import PGGANGenerator
from models.stylegan_generator import StyleGANGenerator
from utils.logger import setup_logger
from utils.manipulator import linear_interpolate
from my_interpolation import my_linear_interpolate
from collections import defaultdict
#from search_interpolation import my_linear_interpolate

def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(
            description='Edit image synthesis with given semantic boundary.')
    parser.add_argument('--metric', type=str, required=True,
                                            choices=['ppl', 'prediction', 'taylor_approximation'])
    parser.add_argument('-m', '--model_name', type=str, required=True,
                                            choices=list(MODEL_POOL),
                                            help='Name of the model for generation. (required)')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                                            help='Directory to save the output results. (required)')
    parser.add_argument('-b', '--boundary_path', type=str, required=True,
                                            help='Path to the semantic boundary. (required)')
    parser.add_argument('-i', '--input_latent_codes_path', type=str, default='',
                                            help='If specified, will load latent codes from given '
                                                     'path instead of randomly sampling. (optional)')
    parser.add_argument('-n', '--num', type=int, default=1000,
                                            help='Number of images for editing. This field will be '
                                                     'ignored if `input_latent_codes_path` is specified. '
                                                     '(default: 1)')
    parser.add_argument('-attr_num', type=int, default=16)
    parser.add_argument('-s', '--latent_space_type', type=str, default='z',
                                            choices=['z', 'Z', 'w', 'W', 'wp', 'wP', 'Wp', 'WP'],
                                            help='Latent space used in Style GAN. (default: `Z`)')
    parser.add_argument('-a', '--attr_index', type=int, required=True,
                                            help='Attribute index to measure on (required)')
    parser.add_argument('--start_distance', type=float, default=-3.0,
                                            help='Start point for manipulation in latent space. '
                                                     '(default: -3.0)')
    parser.add_argument('--end_distance', type=float, default=3.0,
                                            help='End point for manipulation in latent space. '
                                                     '(default: 3.0)')
    parser.add_argument('--steps', type=int, default=600,
                                            help='Number of steps for image editing. (default: 10)')
    parser.add_argument('--step_size', type=float, default=1e-2)
    parser.add_argument('--method', required=True, type=str, choices=['interfacegan', 'ours', 'static'],
                                            help='Type of method to evaluate peceptual length')
    parser.add_argument('--condition', action='store_true',
                                            help='Use conditional constraints or not')
    parser.add_argument('--pretrained_clf_path', type=str, 
                                            default='/home/u5397696/interpolation/celebA-hq-classifier')
    parser.add_argument('--save_results', action='store_true')
    return parser.parse_args()


def process_bound_path(boundary_path, attr_index, gan_type, condition):
    bound_choices = defaultdict(list)
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
        bound_choices[0] = 'stylegan_celebahq_eyeglasses_c_all_boundary.npy'
        bound_choices[1] = 'stylegan_celebahq_gender_c_all_boundary.npy'
        bound_choices[2] = 'stylegan_celebahq_smile_c_all_boundary.npy'
        bound_choices[3] = 'stylegan_celebahq_age_c_all_boundary.npy'
        bound_choices[5] = 'stylegan_celebahq_bald_c_all_boundary.npy'
        bound_choices[7] = 'stylegan_celebahq_narrow_eyes_c_all_boundary.npy'
        bound_choices[10] = 'stylegan_celebahq_blonde_hair_c_all_boundary.npy'
        bound_choices[13] = 'stylegan_celebahq_pale_skin_c_all_boundary.npy'
    else:
        raise ValueError('process_bound_path: unknown gan type.')

    return os.path.join(boundary_path, bound_choices[attr_index])

def simple_interpolation(start, end, coef):
    return coef * start + (1-coef) * end

def generate_segments(interpolations, epsilon=1e-4, samples=100):
    # Plan to sample 1000 trajectories, and generate 100 samples from each trajectory,
    # leading to 100000 images in total
    steps = len(interpolations)
    out = []
    rand_sample = np.random.randint(steps-1, size=samples)
    t = np.random.uniform(size=samples)
    for i in range(samples):
        out1 = simple_interpolation(interpolations[rand_sample[i]], interpolations[rand_sample[i]+1], t[i])
        out2 = simple_interpolation(interpolations[rand_sample[i]], interpolations[rand_sample[i]+1], t[i]+epsilon)
        out.append(out1) 
        out.append(out2)

    out = np.stack(out, axis=0)
    print(out.shape)
    return out

def preprocess_images(images, normalize=False, rescale=False, is_cuda=True):
    # do something to normalize and convert images to be tensors
    processed_img = images.transpose(0, 3, 1, 2)
    
    processed_img = processed_img/255.
    if normalize:
        processed_img = (processed_img-0.5)/0.5

    processed_img = torch.tensor(processed_img).float()
    if is_cuda:
        processed_img = processed_img.cuda()

    if rescale:
        downscale = nn.Upsample(size=224, mode='bilinear')
        if is_cuda:
            downscale.cuda()
        processed_img = downscale(processed_img)

    return processed_img

def compute_batch_lpips(images, **kwargs):
    loss = kwargs['loss']
    with torch.no_grad():
        images = preprocess_images(images, normalize=True).cuda()
        odd_indices = [i*2+1 for i in range(images.shape[0]//2)]
        even_indices = [i*2 for i in range(images.shape[0]//2)]

        d = loss(images[even_indices], images[odd_indices]).squeeze()
        return d.detach().cpu().numpy()

def compute_batch_prediction(images, **kwargs):
    model = kwargs['model']
    with torch.no_grad():
        images = preprocess_images(images, rescale=True, is_cuda=True).cuda()
        pred = model(images)

        return pred.detach().cpu().numpy()

# +
def summarize(p_len, sample_id, total_num, attr_index, metric, gan_type, args):
    if metric == 'ppl':
        print(np.mean(np.concatenate(p_len, axis=0) / ((1e-4) ** 2)))
    elif metric == 'prediction' or metric == 'taylor_approximation':
        fname = f'{gan_type}_{args.method}_{args.metric}_{args.attr_index}_{args.condition}_{args.steps}'
        fname = os.path.join(args.output_dir, fname)
        out = np.concatenate(p_len, axis=0).reshape(sample_id+1, args.steps, args.attr_num)
        np.save(fname, out)
        print(f'prediction shape {out.shape}, mean: {np.mean(np.mean(out, axis=0), axis=0)} saved in {fname}')
    else:
        raise ValueError('Fail to summarize because of unknown metric.')
        
def save_results(images, p_len, noise_list, interpolation_id, sample_id, gan_type, args):
    '''
    for image in images:
        save_path = os.path.join(args.output_dir,
                                 f'{sample_id:03d}_{interpolation_id:03d}.jpg')
        cv2.imwrite(save_path, image[:, :, ::-1])
        interpolation_id += 1
    '''
    if args.metric == 'prediction' or args.metric == 'taylor_approximation':
        fname = f'{gan_type}_{args.method}_{args.metric}_{args.attr_index}_{args.condition}_{args.steps}'
        fname = os.path.join(args.output_dir, fname)
        
        #out = np.concatenate(p_len, axis=0)
        #np.save(fname, out)
        
        noise_arr = np.concatenate(noise_list, axis=0)
        np.save(fname + '_noise', noise_arr)


# +
def main():
    """Main function."""
    args = parse_args()
    attr_index = args.attr_index
    #logger = setup_logger(args.output_dir, logger_name='generate_data')

    print(f'Initializing generator.')
    gan_type = MODEL_POOL[args.model_name]['gan_type']
    if gan_type == 'pggan':
        model = PGGANGenerator(args.model_name, None)
        kwargs = {}
    elif gan_type == 'stylegan':
        model = StyleGANGenerator(args.model_name, None)
        kwargs = {'latent_space_type': args.latent_space_type}
    else:
        raise NotImplementedError(f'Not implemented GAN type `{gan_type}`!')

    
    args.boundary_path = process_bound_path(args.boundary_path, args.attr_index, gan_type, args.condition)
    print(f'Preparing boundary: {args.boundary_path}.')
    if not os.path.isfile(args.boundary_path):
        raise ValueError(f'Boundary `{args.boundary_path}` does not exist!')
    boundary = np.load(args.boundary_path)
    np.save(os.path.join(args.output_dir, 'boundary.npy'), boundary)

    print(f'Preparing latent codes.')
    if os.path.isfile(args.input_latent_codes_path):
        print(f'  Load latent codes from `{args.input_latent_codes_path}`.')
        latent_codes = np.load(args.input_latent_codes_path)
        if len(latent_codes) > 1:
                latent_codes = np.expand_dims(latent_codes[0], axis=0)
        latent_codes = model.preprocess(latent_codes, **kwargs)
    else:
        print(f'  Sample latent codes randomly.')
        latent_codes = model.easy_sample(args.num, **kwargs)
    np.save(os.path.join(args.output_dir, 'latent_codes.npy'), latent_codes)
    total_num = latent_codes.shape[0]

    print(f'Editing {total_num} samples.')
    p_len = []
    g_len = []
    noise_list = []
    kwargs_metric = {}
    if args.metric == 'ppl':
        loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
        kwargs_metric['loss'] = loss_fn_vgg
        compute_metric = compute_batch_lpips
    elif args.metric == 'prediction' or args.metric == 'taylor_approximation':
        attr_num = args.attr_num
        attr_clf = EvalCompoundResNet(args.pretrained_clf_path)
        attr_clf.cuda()
        attr_clf.eval()
        attr_clf.requires_grad = False
        kwargs_metric['model'] = attr_clf
        compute_metric = compute_batch_prediction
    interpolation_id = 0
    for sample_id in tqdm(range(total_num), leave=False):
        if args.method == 'interfacegan':
            """Original interpolation"""
            starting_latent_code = latent_codes[sample_id:sample_id + 1].reshape(1, -1)
            interpolations = my_linear_interpolate(starting_latent_code,
                                                    attr_index,
                                                    boundary,
                                                    'linear',
                                                    steps=args.steps,
                                                    step_size=args.step_size,
                                                    condition=args.condition,
                                                    gan_type=gan_type)
            
            segments = generate_segments(interpolations) if args.metric == 'ppl' else interpolations
            noise_list.append(segments)
            for interpolations_batch in model.get_batch_inputs(segments):
                if gan_type == 'pggan':
                    outputs = model.easy_synthesize (interpolations_batch)
                elif gan_type == 'stylegan':
                    outputs = model.easy_synthesize(interpolations_batch, **kwargs)

                p_len.append(compute_metric(outputs['image'], **kwargs_metric))
                if args.save_results:
                    save_results(outputs['image'], p_len, noise_list, interpolation_id, sample_id, gan_type, args)
                    interpolation_id += outputs['image'].shape[0]
                print(f'attr_index: {args.attr_index}, sample_id: {sample_id}, len of p :{len(p_len)}')
        
        elif args.method == 'ours':
            # attribute modification
            starting_latent_code = latent_codes[sample_id:sample_id + 1].reshape(1, -1)
            interpolations = my_linear_interpolate(starting_latent_code,
                                                    attr_index,
                                                    boundary,
                                                    'piecewise_linear',
                                                    steps=args.steps,
                                                    step_size=args.step_size, 
                                                    condition=args.condition,
                                                    gan_type=gan_type,
                                                    return_more=(args.metric=='taylor_approximation'))
            if args.metric == 'taylor_approximation':
                interpolations, grad, _ = interpolations[0], interpolations[1], interpolations[2]
                # 40 steps only produce 39 gradients
                grad.append(np.zeros((1, 512)))
                grad = np.concatenate(grad, axis=0)
                g_len.append(grad)
            #interpolations = model.preprocess(interpolations, **kwargs)
            segments = generate_segments(interpolations) if args.metric == 'ppl' else interpolations
            noise_list.append(segments)
            for inter_step, interpolations_batch in enumerate(model.get_batch_inputs(segments)):
                if gan_type == 'pggan':
                    outputs = model.easy_synthesize(interpolations_batch)
                elif gan_type == 'stylegan':
                    outputs = model.easy_synthesize(interpolations_batch, **kwargs)
                
                p_len.append(compute_metric(outputs['image'], **kwargs_metric))
                if args.save_results:
                    save_results(outputs['image'], p_len, noise_list, interpolation_id, sample_id, gan_type, args)
                    interpolation_id += outputs['image'].shape[0]
                print(f'attr_index: {args.attr_index}, sample_id: {sample_id}, len of p :{len(p_len)}')
        
        elif args.method == 'static':
            # attribute modification
            starting_latent_code = latent_codes[sample_id:sample_id + 1].reshape(1, -1)
            interpolations = my_linear_interpolate(starting_latent_code,
                                                    attr_index,
                                                    boundary,
                                                    'static_linear',
                                                    steps=args.steps,
                                                    step_size=args.step_size, 
                                                    condition=args.condition,
                                                    gan_type=gan_type,
                                                    return_more=(args.metric=='taylor_approximation'))
            if args.metric == 'taylor_approximation':
                interpolations, grad, _ = interpolations[0], interpolations[1], interpolations[2]
                g_len.append(grad)
            #interpolations = model.preprocess(interpolations, **kwargs)
            segments = generate_segments(interpolations) if args.metric == 'ppl' else interpolations
            noise_list.append(segments)
            for inter_step, interpolations_batch in enumerate(model.get_batch_inputs(segments)):
                if gan_type == 'pggan':
                    outputs = model.easy_synthesize(interpolations_batch)
                elif gan_type == 'stylegan':
                    outputs = model.easy_synthesize(interpolations_batch, **kwargs)
                
                p_len.append(compute_metric(outputs['image'], **kwargs_metric))
                if args.save_results:
                    save_results(outputs['image'], p_len, noise_list, interpolation_id, sample_id, gan_type, args)
                    interpolation_id += outputs['image'].shape[0]
                print(f'attr_index: {args.attr_index}, sample_id: {sample_id}, len of p :{len(p_len)}')
                
        else:
            raise ValueError('Unknown method type.')
        
        #assert interpolation_id == args.steps
        print(f'  Finished sample {sample_id:3d}.')
        if args.metric == 'taylor_approximation' and args.method == 'ours':
            fname = f'{gan_type}_{args.method}_{args.metric}_{args.attr_index}_{args.condition}_{args.steps}_grad'
            fname = os.path.join(args.output_dir, fname)
            out = np.concatenate(g_len, axis=0).reshape(sample_id+1, args.steps, 512)
            np.save(fname, out)
            print('Gradients saved.')
        if args.metric == 'taylor_approximation' and args.method == 'static':
            fname = f'{gan_type}_{args.method}_{args.metric}_{args.attr_index}_{args.condition}_{args.steps}_grad'
            fname = os.path.join(args.output_dir, fname)
            out = np.concatenate(g_len, axis=0).reshape(sample_id+1, 512)
            np.save(fname, out)
            print('Gradients saved.')
        summarize(p_len, sample_id, total_num, attr_index, args.metric, gan_type, args)
    summarize(p_len, sample_id, total_num, attr_index, args.metric, gan_type, args)
    print(f'Successfully edited {total_num} samples.')


if __name__ == '__main__':
    main()
