import mitsuba as mi
import drjit as dr
mi.set_variant('cuda_ad_rgb') # set variant before import anything else
from mitsuba import ScalarTransform4f as T
from sandbox3d_fd import forward
from utils import pcd_to_mesh, multi_view, compute_image_loss
from matplotlib import pyplot as plt
import numpy as np
from imageio.v3 import imread

n_view = 6
guess = -0.08
errors = []
vels = []

# render final state from nerf
data = np.load('./data/nerf_to_mtpts_frame199.npz', allow_pickle=True)
pos = data['pos']
ply_file = pcd_to_mesh(pos=pos)
images_ref, phis = multi_view(ply_file, n_view=n_view,from_o3d=True, save_prefix='')

for iteration in range(20):
    # forward
    print(f'velocity: {guess:.3f}')
    vels.append(guess)
    pos, mat = forward(guess)
    # generate mesh
    ply_file = pcd_to_mesh(pos=pos, mat=mat, mask_id=3)
    # multi view images
    images, phis = multi_view(ply_file, n_view=n_view, from_o3d=True, save_prefix='')
    
    multiview_loss = []
    # compute loss for different views
    for i, (image, image_ref) in enumerate(zip(images,images_ref)):
        image_loss = compute_image_loss(image, image_ref)
        multiview_loss.append(image_loss)
    print(np.mean(multiview_loss))
    errors.append(np.array(multiview_loss))
    guess += 0.02
# save errors
np.savez('data/results2.npz',vel=np.array(vels), err=np.array(errors))