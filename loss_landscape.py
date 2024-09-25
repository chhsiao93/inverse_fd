import mitsuba as mi
import drjit as dr
mi.set_variant('cuda_ad_rgb') # set variant before import anything else
from mitsuba import ScalarTransform4f as T
from sandbox3d_fd import forward
from utils import pcd_to_mesh, multi_view, compute_image_loss
from matplotlib import pyplot as plt
import numpy as np
from imageio.v3 import imread

n_view = 12
force_control=False
guess = 35.0
mses = []
ssims = []
depths = []
guesses = []
# render final state from nerf
data = np.load('./data/nerf_to_mtpts_frame199.npz', allow_pickle=True)
pos = data['pos']
ply_file = pcd_to_mesh(pos=pos)
images_ref, phis = multi_view(ply_file, n_view=n_view,from_o3d=True, save_prefix='')

for iteration in range(41):
    # forward
    print(f'friction angle: {guess:.3f}')
    guesses.append(guess)
    pos, mat = forward(act_vel=-0.2, f_a=0.0, f_b=0.0, f_c=0.0, force_control=force_control, phi_degree=guess)
    # generate mesh
    ply_file = pcd_to_mesh(pos=pos, mat=mat, mask_id=3)
    # multi view images
    images, phis = multi_view(ply_file, n_view=n_view, from_o3d=True, save_prefix='')
    
    multiview_mse = []
    multiview_ssim = []
    multiview_depth = []
    # compute loss for different views
    for i, (image, image_ref) in enumerate(zip(images,images_ref)):
        mse, ssim, depth = compute_image_loss(image, image_ref)
        multiview_mse.append(mse)
        multiview_ssim.append(ssim)
        multiview_depth.append(depth)
    print(np.mean(multiview_mse), np.mean(multiview_ssim), np.mean(multiview_depth))
    mses.append(np.array(multiview_mse))
    ssims.append(np.array(multiview_ssim))
    depths.append(np.array(multiview_depth))
    guess += 0.5
    if iteration%10==0:
        print(mses, ssims, depths)
        
# save errors
np.savez('data/losslandscape_phi.npz',phi=np.array(guesses), mse=np.array(mses), ssim=np.array(ssims), depth=np.array(depths))