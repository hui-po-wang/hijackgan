# +
import cvxpy as cp
import numpy as np
from sklearn import svm
from models.classifiers import UnifiedRegressor, UnifiedDropoutRegressor, DropoutCompoundClassifier
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from PIL import Image


# -

def solve_projection(tgt_dir, cond_dir, momentum, verbose=False):
    '''Solve constrained optimization to find a vector that is orthogonal to others
        tat_dir: Numpy array. A vector of interest
        cond_dir: List of numpy arrays. Vectors, which will be served as conditions
    '''
    dim = tgt_dir.shape[-1]
    if np.linalg.norm(tgt_dir) < 1e-9:
        c = momentum/(np.linalg.norm(momentum) + 1e-9)
    else:
        c = tgt_dir
    A = np.stack(cond_dir, axis=0)
    b = np.zeros(A.shape[0])
    
    x = cp.Variable(dim)
    prob = cp.Problem(cp.Maximize(c@x),
                 [A @ x == b,
                    cp.norm(x) <= 1])
    
    try:
        prob.solve()
    except:
        prob.solve(solver=cp.SCS)
    
    out = x.value
    out[np.abs(out)<1e-4] = 0

    if np.sum(np.abs(out)) < 1e-9:
        return tgt_dir/(np.linalg.norm(tgt_dir) + 1e-9)
    if verbose:
        return out, (prob.value, prob.status)
    else:
        return out


def code_proj(init_pt, m, attr_index, logit=None):
    """Unconditional projection
    """
    c = init_pt
    if logit is not None:
        pred = logit
    else:
        pred = m(c)
    grad = torch.autograd.grad(pred[:, attr_index], c)[0]
    grad = grad/(torch.norm(grad, p=2, dim=1)+1e-9)
    
    return grad.detach().cpu().numpy()


def cond_code_cvx(init_pt, m, attr_index, invariant_index, momentum, logit=None):
    """conditional projection
    """
    invariant_main = invariant_index['main']
    invariant_others = invariant_index['others']
    m_main = m['main']
    m_others = m['others']
    
    c = init_pt
    if logit is None:
        pred = m_main(c)
    else:
        pred = logit
    tar_grad = torch.autograd.grad(pred[:, attr_index], c)[0]
    
    cond_grad = []    
    if m_others is not None and invariant_others is not None:
        pred = m_others(c)
        for index in invariant_others:
            grad = torch.autograd.grad(pred[:, index], c, retain_graph=True)[0]
            cond_grad.append(grad.squeeze().detach().cpu().numpy())
    
    pred = m_main(c)
    for index in invariant_main:
        grad = torch.autograd.grad(pred[:, index], c, retain_graph=True)[0]
        cond_grad.append(grad.squeeze().detach().cpu().numpy())
        
    out_grad = solve_projection(tar_grad.detach().cpu().numpy(), cond_grad, momentum)
        
    return out_grad


def cond_code_cvx_landmark(init_pt, m, attr_index, invariant_index, attr_invariant, momentum, logit=None):
    """conditional projection
    """
    invariant_main = invariant_index['main']
    invariant_others = invariant_index['others']
    m_main = m['main']
    m_others = m['others']
    m_attr = m['more']
    
    c = init_pt
    if logit is None:
        pred = m_main(c)
    else:
        pred = logit
    tar_grad = torch.autograd.grad(pred[:, attr_index], c)[0]
    
    cond_grad = []    
    if m_others is not None and invariant_others is not None:
        pred = m_others(c)
        for index in invariant_others:
            grad = torch.autograd.grad(pred[:, index], c, retain_graph=True)[0]
            cond_grad.append(grad.squeeze().detach().cpu().numpy())
    
    if m_attr is not None and attr_invariant is not None:
        pred = m_attr(c)
        for index in attr_invariant:
            grad = torch.autograd.grad(pred[:, index], c, retain_graph=True)[0]
            cond_grad.append(grad.squeeze().detach().cpu().numpy())
            
    pred = m_main(c)
    for index in invariant_main:
        grad = torch.autograd.grad(pred[:, index], c, retain_graph=True)[0]
        cond_grad.append(grad.squeeze().detach().cpu().numpy())
        
    out_grad = solve_projection(tar_grad.detach().cpu().numpy(), cond_grad, momentum)
        
    return out_grad


def cond_code_proj(init_pt, m, attr_index, invariant_index):
    """conditional projection
    """
    
    invariant_main = invariant_index['main']
    invariant_others = invariant_index['others']
    m_main = m['main']
    m_others = m['others']
    
    c = init_pt
    pred = m_main(c)
    tar_grad = torch.autograd.grad(pred[:, attr_index], c)[0]
    tar_grad = tar_grad/torch.norm(tar_grad, p=2, dim=1)
    
    if m_others is not None and invariant_others is not None:
        for index in invariant_others:
            pred = m_others(c)
            grad = torch.autograd.grad(pred[:, index], c)[0]
            grad = grad/torch.norm(grad, p=2, dim=1)
            tar_grad = tar_grad - torch.dot(tar_grad.squeeze(), grad.squeeze())*grad
            tar_grad = tar_grad/torch.norm(tar_grad, p=2, dim=1)
            
    for index in invariant_main:
        pred = m_main(c)
        grad = torch.autograd.grad(pred[:, index], c)[0]
        grad = grad/torch.norm(grad, p=2, dim=1)
        tar_grad = tar_grad - torch.dot(tar_grad.squeeze(), grad.squeeze())*grad
        tar_grad = tar_grad/torch.norm(tar_grad, p=2, dim=1)
        
    return tar_grad


def poly_lr_scheduler(base_lr, iter, max_iter=30000, power=0.9):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


# +
def pose_edit(latent_code,
                 attr_index,
                 model,
                 steps,
                 gan_type,
                 condition,
                 direction):
    
    if len(latent_code.shape) == 2:
        crit = nn.MSELoss(reduction='none')
        out = []
        out_grad = []
        out_stepsize = []
        
        c = torch.tensor(latent_code).float().cuda()
        c.requires_grad = True
        
        m_main = model['main']
        m_others = model['others']
        step_size = model['step_size']
        
        momentum = np.random.randn(512)
        beta = 0.9
        for i in tqdm(range(steps)):
            if i == 0:
                if direction is not None:
                    sign = direction
                else:
                    sign = -1
                out.append(c.detach().cpu().numpy())

                invariant_index = generate_invariant_matrix(attr_index, is_pose=True)
                continue
            
            
            pred = m_main(c)
            
            if condition:
                grad_dir = sign * cond_code_cvx(c, model, attr_index, invariant_index, momentum)
            else:
                #grad_dir = sign * code_proj(c, m_main, attr_index, logit=loss)
                grad_dir = sign * code_proj(c, m_main, attr_index)
                
            c = c.detach().cpu().numpy()
            if np.linalg.norm(grad_dir) == 0:
                c = c + step_size * 0.1 * momentum/(np.linalg.norm(momentum) + 1e-9)
            else:
                c = c + step_size * grad_dir

            if steps >= 50 and i % (steps//50) == 0:
                out.append(c)
            out_grad.append(grad_dir)
            out_stepsize.append(step_size)
            
            c = torch.tensor(c).cuda().float()
            c.requires_grad = True
            
        out = np.concatenate(out, axis=0)
        return out, out_grad, out_stepsize
    raise ValueError(f'Input `latent_code` should be with shape '
                     f'[1, latent_space_dim] or [1, N, latent_space_dim] for '
                     f'W+ space in Style GAN!\n'
                     f'But {latent_code.shape} is received.')


def piecewise_linear_interpolate(latent_code,
                         attr_index,
                         model,
                         steps,
                         is_landmark,
                         adaptive,
                         gan_type,
                         condition,
                         direction):
    
    if len(latent_code.shape) == 2:
        out = []
        out_grad = []
        out_stepsize = []
        #if gan_type == 'stylegan':
        #    latent_code = preprocess(latent_code)
        c = torch.tensor(latent_code).float().cuda()
        c.requires_grad = True
        
        m_main = model['main']
        m_others = model['others']
        if is_landmark:
            m_attr = model['more']
            attr_invariant = [1, 10, 13]
        step_size = model['step_size']
        
        momentum = np.random.randn(512)
        beta = 0.9
        for i in tqdm(range(steps)):
            if i == 0:
                if not is_landmark:
                    pred = m_main(c)
                    pred = torch.sigmoid(pred)[:, attr_index].detach().cpu().numpy()
                    sign = -1 if pred > 0.5 else 1
                    #sign *= -1
                    out.append(c.detach().cpu().numpy())
                    
                    invariant_index = generate_invariant_matrix(attr_index, is_landmark=False)
                else:
                    if direction is not None:
                        sign = direction
                    else:
                        sign = -1
                    #sign *= -1
                    out.append(c.detach().cpu().numpy())
                    
                    invariant_index = generate_invariant_matrix(attr_index, is_landmark=True)
                continue
                
            pred = m_main(c)
            if condition:
                if is_landmark:
                    grad_dir = sign * cond_code_cvx_landmark(c, model, attr_index, invariant_index, attr_invariant, momentum)
                else:
                    grad_dir = sign * cond_code_cvx(c, model, attr_index, invariant_index, momentum)
            else:
                grad_dir = sign * code_proj(c, m_main, attr_index)
            momentum = momentum + sign * code_proj(c, m_main, attr_index)    
                
            c = c.detach().cpu().numpy()
            if adaptive:
                step_size = binary_search_stepsize(c, grad_dir, model, invariant_index)
                c = c + step_size * grad_dir + beta * momemtum
            else:
                if np.linalg.norm(grad_dir) == 0:
                    c = c + step_size * 0.1 * momentum/(np.linalg.norm(momentum) + 1e-9)
                else:
                    c = c + step_size * grad_dir
                    
            if not is_landmark:
                out.append(c)
            #elif is_landmark and i % (steps//40) == 0:
            else:
                out.append(c)
            out_grad.append(grad_dir/sign)
            out_stepsize.append(step_size)
            
            c = torch.tensor(c).cuda().float()
            c.requires_grad = True
        out = np.concatenate(out, axis=0)
        
        return out, out_grad, out_stepsize
    raise ValueError(f'Input `latent_code` should be with shape '
                     f'[1, latent_space_dim] or [1, N, latent_space_dim] for '
                     f'W+ space in Style GAN!\n'
                     f'But {latent_code.shape} is received.')


def generate_invariant_matrix(attr_index, is_landmark=False, is_pose=False):
    output_index = {'main': None, 'others':None}
    if is_pose:
        #output_index['main'] = [num for num in range(2) if num != attr_index]
        output_index['main'] = [1]
        output_index['others'] = [1, 10, 13]
    elif is_landmark:
        if attr_index in [0, 1]:
            landmark_index = list(range(4, 10))
        elif attr_index in [6, 7]:
            landmark_index = [4, 5, 8, 9]
        elif attr_index in [8, 9]:
            landmark_index = [4, 5, 6, 7]
        elif attr_index in [4, 5]:
            landmark_index = [4, 5]
            landmark_index.remove(attr_index)
        else:
            landmark_index = [num for num in range(10) if num != attr_index]
            
        output_index['main'] = landmark_index
        # Note that when modifying mouth landmarks, the smlie should not be fixed
        output_index['others'] = [num for num in range(2)]
    else:
        output_index['main'] = [num for num in range(4) if num != attr_index]
    
    return output_index



def baseline_linear_interpolate(latent_code,
                         attr_index,
                         boundary,
                         model,
                         steps):
    assert (latent_code.shape[0] == 1 and boundary.shape[0] == 1 and
            len(boundary.shape) == 2 and
            boundary.shape[1] == latent_code.shape[-1])
    
    attr_clf = model['main']
    step_size = model['step_size']
    dist = latent_code.dot(boundary.T)
    linspace = np.linspace(0, 3+np.abs(dist), steps)
    pred = torch.sigmoid(attr_clf(torch.tensor(latent_code).float().cuda()))
    sign = 1 if (pred[0, attr_index]).detach().cpu().numpy() < 0.5 else -1
    #sign *= -1
    if attr_index == 3:
        sign *= -1
    
    if len(latent_code.shape) == 2:
        out = []
        #step_size = 1e-2
        edited_code = latent_code
        for i in range(steps):
            edited_code = edited_code + sign * step_size * boundary
            out.append(edited_code)
        return np.concatenate(out, axis=0)

    if len(latent_code.shape) == 3:
        linspace = linspace.reshape(-1, 1, 1).astype(np.float32)
        return latent_code + sign * linspace * boundary.reshape(1, 1, -1)
    raise ValueError(f'Input `latent_code` should be with shape '
                     f'[1, latent_space_dim] or [1, N, latent_space_dim] for '
                     f'W+ space in Style GAN!\n'
                     f'But {latent_code.shape} is received.')



def static_linear_interpolate(latent_code,
                         attr_index,
                         boundary,
                         model,
                         steps):
    assert (latent_code.shape[0] == 1 and boundary.shape[0] == 1 and
            len(boundary.shape) == 2 and
            boundary.shape[1] == latent_code.shape[-1])
    
    m_main = model['main']
    step_size = model['step_size']
    
    if len(latent_code.shape) == 2:
        out = []

        edited_code = latent_code
        pt_tensor = torch.tensor(latent_code).float().cuda()
        pt_tensor.requires_grad = True

        pred = torch.sigmoid(m_main(pt_tensor))[:, attr_index].detach().cpu().numpy()
        sign = -1 if pred > 0.5 else 1
        grad = code_proj(pt_tensor, m_main, attr_index, logit=None)
        for i in range(steps):
            edited_code = edited_code + sign * step_size * grad
            out.append(edited_code)
        return np.concatenate(out, axis=0), grad, step_size

    if len(latent_code.shape) == 3:
        linspace = linspace.reshape(-1, 1, 1).astype(np.float32)
        return latent_code + sign * linspace * boundary.reshape(1, 1, -1)
    raise ValueError(f'Input `latent_code` should be with shape '
                     f'[1, latent_space_dim] or [1, N, latent_space_dim] for '
                     f'W+ space in Style GAN!\n'
                     f'But {latent_code.shape} is received.')



def interpolate(latent_code,
                         attr_index,
                         boundary,
                         mode,
                         steps=10,
                         is_landmark=False,
                         adaptive=False,
                         condition=True,
                         save_trajectory=False,
                         gan_type='stylegan',
                         step_size=.6,
                         return_more=False,
                         direction=None):
    
    m_landmark = None
    if gan_type == 'stylegan':
        print (f'Loading classifiers for StyleGAN. Condition: {condition}.')
        m_landmark = UnifiedRegressor(10) 
        m_landmark.load_state_dict(torch.load('./models/pretrain/stylegan_landmark_z.pt'))
        
        m_pose = UnifiedDropoutRegressor(3)
        m_pose.load_state_dict(torch.load('./models/pretrain/stylegan_headpose_z_dp.pt'))
        
        m_attr = DropoutCompoundClassifier(16)
        m_attr.load_state_dict(torch.load('./models/pretrain/stylegan_celebahq_z.pt'))
        
        m_landmark.eval()
        m_landmark.cuda()
        
        m_pose.eval()
        m_pose.cuda()
    elif gan_type == 'pggan':
        print (f'Loading classifiers for PGGAN. Condition: {condition}.')
        m_attr = DropoutCompoundClassifier(16)
        m_attr.load_state_dict(torch.load('./models/pretrain/pggan_celebahq_z.pt'))
        
    m_attr.eval()
    m_attr.cuda()    
    
    model = {'main': None, 'others': None, 'step_size': step_size, 'more': None}
    
    if mode == 'linear':
        model['main'] = m_attr
        return baseline_linear_interpolate(latent_code, attr_index, boundary, model, steps)
    elif mode == 'piecewise_linear':
        if is_landmark:
            model['main'] = m_landmark
            model['others'] = m_pose
            model['more'] = m_attr
        else:
            model['main'] = m_attr
            model['others'] = m_landmark
        code, grads, steps_size = piecewise_linear_interpolate(latent_code,
                         attr_index,
                         model,
                         steps,
                         is_landmark,
                         adaptive,
                         gan_type,
                         condition,
                         direction)
    elif mode == 'pose_edit':
        model['main'] = m_pose
        model['others'] = m_attr
        code, grads, steps_size = pose_edit(latent_code,
                         attr_index,
                         model,
                         steps,
                         gan_type,
                         condition,
                         direction)
    elif mode == 'static_linear':
        model['main'] = m_attr
        code, grads, step_size =  static_linear_interpolate(latent_code, attr_index, boundary, model, steps)
    else:
        raise ValueError(f'Unknown type {mode} received.')
    
    # TODO: if save_trajectory is true, save code, grads, step_size
    #print(f'final angle: {np.dot(grads[0], grads[-1].T)}')
    np.save('./test_code.npy', code)
    if return_more:
        return code, grads, step_size
    else:
        return code
