`python run_inverse.py` will run 5 times of sandbox-actuator simulation with increasing initial actuator velocity. In each iteraction, it will use the last step of particle position to render the scene from 6 differnt views. It will compare the images with the reference image and compute the image loss. The reference images `data/image_refX.png` are rendered from `data/nerf_to_mtpts_frame199.npz` which is the point cloud from NeRF.

`python plot_result.py` will plot the loss vs the initial actuator velocity for each view.
