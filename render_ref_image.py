import mitsuba as mi
import drjit as dr
mi.set_variant('cuda_ad_rgb') # set variant before import anything else
from mitsuba import ScalarTransform4f as T
from utils import pcd_to_mesh, multi_view
from matplotlib import pyplot as plt
import numpy as np
from imageio.v3 import imread

n_view = 12
# render final state from nerf
data = np.load('./data/nerf_to_mtpts_frame199.npz', allow_pickle=True)
pos = data['pos']
ply_file = pcd_to_mesh(pos=pos)
images, phis = multi_view(ply_file, n_view=n_view,from_o3d=True, save_prefix='')
images_from_mi = np.array(images)
for i, image in enumerate(images):
        mi.util.write_bitmap(f"data/image_ref{i:02d}.png", image[:,:,:3])
