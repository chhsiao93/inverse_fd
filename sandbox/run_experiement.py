import numpy as np
import open3d as o3d
import alphashape
from tqdm import tqdm

from pathlib import Path
import sys
# Add the simulation/ directory to Python's module search path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from sandbox_simulation import forward
from utils import snapshot, update_xml


scene='sandbox'
friction_angle=45

format = 'mesh'  # 'pcd' or 'mesh'
alpha = 20  # alpha for mesh generation
show = False

npz_path = 'data/sand_particles.npz'  # Path to the npz file with sand particles from MPM
data_path = f'data/sandbox_experiment{friction_angle}.npz'
# check if data already exists
if Path(data_path).exists():
    print(f"Data already exists at {data_path}. Loading...")
    # if data is already generated, load it
    print(f"Loading data from {data_path}")
    data = np.load(data_path, allow_pickle=True)
    pos = data['pos']
    mat = data['mat']
    
else:
    print(f"Data not found at {data_path}. Running simulation...")
    pos, mat, _ = forward(phi_degree=friction_angle, sand_density=1.0, npz_path=npz_path)
    print(f"Simulation completed with friction angle: {friction_angle} degrees")
    # save trajectory
    np.savez(f'data/sandbox_experiment{friction_angle}.npz', pos=pos, mat=mat, stress=stress)
    print(f"Data saved to data/sandbox_experiment{friction_angle}.npz")



sand_pos = pos[:, mat == 3, :]  # Extract sand particles
actuator_pos = pos[:, mat == 5, :]  # Extract actuator particles

output_dir = 'mesh'

# generate mesh for all frames
n_frames = pos.shape[0]
frames = range(n_frames)
observe_frames = [50, 100, 150, 199]

for frame in observe_frames:
    # making point cloud
    pcd = o3d.geometry.PointCloud() 
    pcd.points = o3d.utility.Vector3dVector(sand_pos[frame])

    # save sand mesh

    mesh_sand = alphashape.alphashape(np.asarray(pcd.points), alpha)
    mesh_sand.export(f"{output_dir}/sand{frame:04d}.ply")
            
    # save actuator mesh 
    corners = actuator_pos[frame].reshape(100, 100, 3)
    corner1 = corners[0, 0]
    corner2 = corners[0, -1]
    corner3 = corners[-1, -1]
    corner4 = corners[-1, 0]
    plane = o3d.geometry.TriangleMesh()
    plane.vertices = o3d.utility.Vector3dVector([corner1, corner2, corner3, corner4])
    plane.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
    plane.compute_vertex_normals()
    o3d.io.write_triangle_mesh(f"{output_dir}/plow{frame:04d}.ply", plane)


# save rendered image
import mitsuba as mi
import drjit as dr
mi.set_variant('cuda_ad_rgb')
import matplotlib.pyplot as plt
observe_frames = [50, 100, 150, 199]
img_folder = "observations/"
spp = 128
xml_file = "scene.xml"
for i, frame in enumerate(observe_frames):
    update_xml(xml_file=xml_file, id='sand_traj', value=f"mesh/sand{frame:04d}.ply")
    update_xml(xml_file=xml_file, id='plow_traj', value=f"mesh/plow{frame:04d}.ply")
    scene = mi.load_file(xml_file)
    left_img = snapshot(scene, sensor=1, spp=spp)
    mi.util.write_bitmap(f"{img_folder}/frame_{frame:04d}_left.png", left_img)
    right_img = snapshot(scene, sensor=0, spp=spp)
    mi.util.write_bitmap(f"{img_folder}/frame_{frame:04d}_right.png", right_img)