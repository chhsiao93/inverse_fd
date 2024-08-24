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
n_view = 6
guess_0 = -0.1
vels = []
errors = []

# def callbackF(x):
#     global n_iter
#     print(f'{n_iter}\t{x[0]:.2e}\t{f(x[0]):.2e}')
#     n_iter += 1
# render final state from nerf
data = np.load('./data/nerf_to_mtpts_frame199.npz', allow_pickle=True)
pos = data['pos']
ply_file = pcd_to_mesh(pos=pos)
images_ref, phis = multi_view(ply_file, n_view=n_view,from_o3d=True, save_prefix='')

    
def f(guess):
    # forward
    guess_scalar = guess.squeeze()
    print(f'velocity: {guess_scalar:.3f}')
    vels.append(guess)
    pos, mat = forward(guess_scalar)
    # generate mesh
    ply_file = pcd_to_mesh(pos=pos, mat=mat, mask_id=3)
    # multi view images
    images, phis = multi_view(ply_file, n_view=n_view, from_o3d=True, save_prefix='')
    
    multiview_loss = []
    # compute loss for different views
    for i, (image, image_ref) in enumerate(zip(images,images_ref)):
        image_loss = compute_image_loss(image, image_ref)
        multiview_loss.append(image_loss)
    errors.append(np.array(multiview_loss))
    print(np.mean(multiview_loss))
    return (np.mean(multiview_loss))


# import numpy as np
# def f(x):
#     print(x)
#     return x**4 - 3*x**3 + 20*x**2 - 5

res = minimize(f, guess_0, method='BFGS', tol=1e-6, options={'maxiter':20})
print('optimized x', res.x)
# save errors
np.savez('data/bfgs_results.npz',vel=np.array(vels), err=np.array(errors))

