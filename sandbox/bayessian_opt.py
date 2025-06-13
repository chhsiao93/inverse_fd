import mitsuba as mi
import drjit as dr
mi.set_variant('cuda_ad_rgb')
from mitsuba import ScalarTransform4f as T
import numpy as np
import matplotlib.pyplot as plt
from skopt import gp_minimize

import yaml
import pickle
from pathlib import Path
import sys
# Add the simulation/ directory to Python's module search path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from sandbox_simulation import forward
from utils import pcd_to_mesh, compute_image_loss, snapshot, update_xml


# ---------- Load Config ----------
def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config("config.yaml")

# ---------- Unpack Config ----------
output_name = config["output_name"]
var_name = config["var_name"]
var_bound = config["var_bound"]
var_bound = [tuple(bound) for bound in config["var_bound"]]
loss_name = config["loss_name"]
acq_func = config["acq_func"]
n_random_starts = config["n_random_starts"]
n_calls = config["n_calls"]
num_bo = config["num_bo"] # Number of Bayesian Optimization iterations
xml_file = config["xml_file"]
npz_path = config["npz_path"]
observe_frames = config["observe_frames"]
observe_image_folder = config["observe_image_folder"]



# ---------- Load Reference Images ----------
ref_images = []
for frame in observe_frames:
    left_img = plt.imread(f"{observe_image_folder}/frame_{frame:04d}_left.png")[:, :, :3]
    right_img = plt.imread(f"{observe_image_folder}/frame_{frame:04d}_right.png")[:, :, :3]
    ref_images.extend([left_img, right_img])

# ---------- Objective Function ----------
def objective(x):
    print(f"Evaluating at x = {x}")

    if var_name == 'density':
        pos, mat, stress = forward(phi_degree=45.0, sand_density=x[0], npz_path=npz_path)
    elif var_name == 'phi':
        pos, mat, stress = forward(phi_degree=x[0], sand_density=1.0, npz_path=npz_path)
    elif var_name == 'both':
        pos, mat, stress = forward(phi_degree=x[0], sand_density=x[1], npz_path=npz_path)
    else:
        raise ValueError("Unrecognized var_name. Use 'phi', 'density', or 'both'.")

    pos = pos[observe_frames, :][:, mat == 3]
    images = []

    for i, frame in enumerate(observe_frames):
        # Convert material points to mesh and save it
        mesh_path = Path(f"mesh/sand_reconstr{frame:04d}.ply")
        pcd_to_mesh(pos[i], str(mesh_path))
        # Update XML file with the new mesh path
        update_xml(xml_file, id='sand_traj', value=str(mesh_path))
        update_xml(xml_file, id='plow_traj', value=f"mesh/plow{frame:04d}.ply")
        # Render the scene
        scene = mi.load_file(xml_file)
        images.extend([snapshot(scene, sensor=1), snapshot(scene, sensor=0)])

    print("Rendered images complete.")

    losses = [compute_image_loss(img, ref)[0] for img, ref in zip(images, ref_images)]
    mean_loss = np.mean(losses)
    print(f"Mean Multiview Loss: {mean_loss:.4e}")
    return mean_loss


for bo_iter in range(num_bo):
    # ---------- Bayesian Optimization ----------
    res = gp_minimize(
        func=objective,
        dimensions=var_bound,
        acq_func=acq_func,
        n_calls=n_calls,
        n_random_starts=n_random_starts,
        noise=0,
    )

    # ---------- Save Result ----------
    result_path = Path(f"result/{output_name}_{bo_iter}.pkl")
    result_path.parent.mkdir(parents=True, exist_ok=True)

    with open(result_path, 'wb') as f:
        pickle.dump(res, f)

    print(f"Optimization complete. Results saved to {result_path}")
    print(f"Optimized {var_name}: {res.x}")
    print(f"Estimated loss: {res.fun:.4e}")
