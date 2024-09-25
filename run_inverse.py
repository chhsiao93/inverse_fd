import mitsuba as mi
import drjit as dr
mi.set_variant('cuda_ad_rgb') # set variant before import anything else
from mitsuba import ScalarTransform4f as T
from sandbox3d_fd import forward
from utils import pcd_to_mesh, multi_view, compute_image_loss
from matplotlib import pyplot as plt
import numpy as np
from imageio.v3 import imread
from scipy.optimize import minimize
n_iter = 0
n_view = 12
guess_0 = [38.5]
guesses = []
errors = []

# render final state from nerf
data = np.load('./data/nerf_to_mtpts_frame199.npz', allow_pickle=True)
pos = data['pos']
ply_file = pcd_to_mesh(pos=pos)
images_ref, phis = multi_view(ply_file, n_view=n_view,from_o3d=True, save_prefix='')

    
def f(guess):
    # forward
    print(guess)
    guesses.append(guess)
    pos, mat = forward(act_vel=-0.2, phi_degree=guess[0])
    # generate mesh
    ply_file = pcd_to_mesh(pos=pos, mat=mat, mask_id=3)
    # multi view images
    images, phis = multi_view(ply_file, n_view=n_view, from_o3d=True, save_prefix='')
    
    multiview_loss = []
    # compute loss for different views
    for i, (image, image_ref) in enumerate(zip(images,images_ref)):
        mse_loss, _, _ = compute_image_loss(image, image_ref)
        multiview_loss.append(mse_loss)
    errors.append(np.array(multiview_loss))
    print(np.mean(multiview_loss))
    return (np.mean(multiview_loss))


# import numpy as np
# def f(x):
#     print(x)
#     return x**4 - 3*x**3 + 20*x**2 + np.sin(10*x)*1e-4

method = 'Nelder-Mead' #Nelder-Mead
tol = 2.0e-4
res = minimize(f, guess_0, method=method, tol=tol, options={'xatol':1e-4,'disp':True})
# res = minimize(f, guess_0, method=method, tol=tol, options={'disp':True,'eps':1e-3})
print('optimized x', res.x)
# save errors
np.savez(f'data/phi_{method}_results.npz',guess=np.array(guesses), err=np.array(errors))

