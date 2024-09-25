import mitsuba as mi
import drjit as dr
mi.set_variant('cuda_ad_rgb') # set variant before import anything else
from mitsuba import ScalarTransform4f as T
from sandbox_simulation import forward
from utils import pcd_to_mesh, multi_view, compute_image_loss
from matplotlib import pyplot as plt
import numpy as np
from imageio.v3 import imread
from skopt import gp_minimize
from scipy.optimize import minimize
import pickle 
from matplotlib import pyplot as plt

output_name = 'test'
n_view = 12 # number of camera view to compute loss
var_name = 'phi' # varible to be optimized. e.g., 'phi' for sand friction angle, 'vel' for actuator velocity.
var_bound = [(35.0, 55.0)] # boundary of optimized varible
loss_name = 'mse' # loss function to be minized. e.g., 'mse' for mean squared error of RGB, 'ssim' for negative structural similarity index, 'depth' for mean squared error of depth.
acq_func = 'EI' # acquisition function for Bayesian opt
n_random_starts = 5 # initialize Gaussian prior with 5 random evaluations
n_calls = 15 # total evaluation in Bayesian, including random start.

# render final state from nerf
data = np.load('./data/nerf_to_mtpts_frame199.npz', allow_pickle=True)
pos = data['pos']
ply_file = pcd_to_mesh(pos=pos)
images_ref, _ = multi_view(ply_file, n_view=n_view,from_o3d=True, save_prefix='')

# the function to be optimized: Foward
def f(x):
    # print(x)
    # forward
    if var_name=='vel':
        pos, mat = forward(act_vel=x[0])
    elif var_name=='phi':
        pos, mat = forward(act_vel=-0.2, phi_degree=x[0])
    else:
        print("unrecognized var_name, use 'phi' or 'vel'")
        return
    # generate mesh
    ply_file = pcd_to_mesh(pos=pos, mat=mat, mask_id=3)
    # multi view images
    images, _ = multi_view(ply_file, n_view=n_view, from_o3d=True, save_prefix='')
    
    multiview_loss = []
    # compute loss for different views
    for i, (image, image_ref) in enumerate(zip(images,images_ref)):
        mse_loss, ssim_value, depth_loss = compute_image_loss(image, image_ref)
        if loss_name == 'depth':
            multiview_loss.append(depth_loss)
        elif loss_name == 'ssim':
            multiview_loss.append(-1.0*ssim_value)
        else:
            multiview_loss.append(mse_loss)
    mean_multiview_loss = np.mean(multiview_loss)
    print(f'Evaluating {var_name} at {x[0]:.4f}, the mean multiview loss is {mean_multiview_loss}')
    return mean_multiview_loss

# running Bayesian Optimization
res = gp_minimize(f,                  # the function to minimize
                  var_bound,      # the bounds on each dimension of x
                  acq_func=acq_func,      # the acquisition function
                  n_calls=n_calls,         # the number of evaluations of f
                  n_random_starts=n_random_starts,  # the number of random initialization points
                  noise=0,       # the noise level (optional)
                  random_state=1234)   # the random seed

print(f'Opimtized {var_name} is at {res.x}, and the estimate loss is {res.fun}')

with open(f'result/{output_name}.pkl', 'wb') as file:
    pickle.dump(res, file)
    print(f'The result is saved as result/{output_name}.pkl')
    
