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
guess = -0.1
# read ref images. the value is integer [0, 255]. Convert it to float [0, 1]
# images_ref = [mi.TensorXf(mi.Bitmap(f'data/image_ref{i:02d}.png'))/255.0 for i in range(n_view)]
images_ref = [(imread(f'data/image_ref{i:02d}.png'))/255.0 for i in range(n_view)]
images_ref = np.array(images_ref)
errors = []
vels = []
for iteration in range(5):
    # forward
    print(f'velocity: {guess:.3f}')
    vels.append(guess)
    pos, mat = forward(guess)
    # generate mesh
    ply_file = pcd_to_mesh(pos=pos, mat=mat, mask_id=3)
    # multi view images
    images, phis = multi_view(ply_file, n_view=n_view, from_o3d=True, save_prefix='')
    # save images
    multiview_loss = []
    for i, image in enumerate(images):
        mi.util.write_bitmap(f"data/image_{i:02d}.png", image)

    images_pred = [(imread(f'data/image_{i:02d}.png'))/255.0 for i in range(n_view)]    
    images_pred = np.array(images_pred)
    
    # compute loss for different views
    for i, (image, image_ref) in enumerate(zip(images_pred,images_ref)):
        image_loss = compute_image_loss(image, image_ref)
        multiview_loss.append(image_loss)
    print(np.mean(multiview_loss))
    errors.append(np.array(multiview_loss))
    guess += -0.03
# save errors
np.savez('data/results.npz',vel=np.array(vels), err=np.array(errors))