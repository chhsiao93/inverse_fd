import numpy as np
import open3d as o3d
from skimage.metrics import structural_similarity as ssim
import trimesh
import alphashape
import xml.etree.ElementTree as ET
import drjit as dr
import mitsuba as mi

# edit the xml file directly
def snapshot(scene, spp=128, sensor=0, linear=False):
    """Render a snapshot of the scene."""
    image = mi.render(scene, spp=spp, sensor=sensor)
    if linear:
        image = np.array(image)
        image = np.where(image <= 0.0031308, 12.92 * image, 1.055 * (image ** (1.0 / 2.4)) - 0.055)
        # Clip values to ensure they are in the valid range [0,1]
        image = np.clip(image, 0, 1)

    return image

def update_xml(xml_file, id, value, save_file=None,):
    """Update frame parameter in the XML file."""
    # Load and parse the XML file
    xml_file = xml_file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Modify the XML file to change the "value" for "filename" of "sand_traj"
    for shape in root.findall(f".//shape[@id='{id}']"):
        for element in shape.findall("string"):
            if element.attrib.get('name') == "filename":
                element.attrib['value'] = value
    if save_file is None:
        save_file = xml_file
    # Save the modified XML file
    tree.write(save_file)

def pcd_to_mesh(pos, save_name='mesh.ply', alpha=20.0):
    
    # Create a point cloud for sand
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pos)
    mesh_sand = alphashape.alphashape(np.asarray(pcd.points), alpha)
    mesh_sand.export(save_name)
    
    return save_name

def compute_image_loss(image, image_ref):
    # convert to numpy array
    image = np.array(image)
    image_ref = np.array(image_ref)
    if image.shape[-1] == 3: #rgb
        mse_loss = (np.mean((image - image_ref)**2))
        ssim_value = ssim(np.array(image_ref), np.array(image), data_range=np.max(image_ref) - np.min(image_ref), channel_axis=2)
        return mse_loss, ssim_value
    elif image.shape[-1] == 5: #rgba and depth
        # ignore a
        image_rgb, image_ref_rgb = image[:,:,:3], image_ref[:,:,:3] #rgb 
        image_depth, image_ref_depth = image[:,:,-1], image_ref[:,:,-1] #depth
        
        mse_loss = (np.mean((image_rgb - image_ref_rgb)**2))
        ssim_value = ssim(np.array(image_ref_rgb), np.array(image_rgb), data_range=np.max(image_ref_rgb) - np.min(image_ref_rgb), channel_axis=2)
        mse_depth_loss = (np.mean((image_depth - image_ref_depth)**2))
        
        return mse_loss, ssim_value, mse_depth_loss
    

def cube(lower_corner, cube_size, dx=1/32, density=2**3):
    cube_vol = cube_size[0] * cube_size[1] * cube_size[2]
    cell_per_cube = cube_vol / (dx ** 3) # number of cells in a cube
    num_particles = int(cell_per_cube * density + 1) # +1 to avoid empty particles
    cube_particles = np.random.rand(num_particles, 3) * cube_size + lower_corner    
    return cube_particles

def generate_mpm_points_from_mesh(
        mesh,
        density=2**3,
        dx=1/32,
        show=False,
        swap_yz=False,
        seed=0):
    np.random.seed(seed)
    # load water_tight mesh
    mesh = trimesh.load_mesh(mesh)
    # check if mesh is watertight
    # assert mesh.is_watertight, "Mesh is not watertight."
    # find bounding box
    world_bound = mesh.bounds
    # generate points based on bounding box
    lower_corner = world_bound[0]
    cube_size = world_bound[1] - world_bound[0]
    points = cube(lower_corner=lower_corner, cube_size=cube_size, density=density, dx=dx)
    # check if points are inside the mesh
    iscontained = mesh.contains(points)
    # filter points
    points = points[iscontained]
    if swap_yz:
        # for taichi mpm space: y is up
        points = points[:, [0, 2, 1]]
    print(f"Lower corner: {lower_corner}")
    print(f"Cube size: {cube_size}")
    print(f"Number of points: {points.shape[0]}")
    
    if show:
        # show origin and xyz axis
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.6, origin=[0.0, 0.0, 0.0])
        # visualize mesh
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        o3d_mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([o3d_mesh, mesh_frame])
        # visualize points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals()
        o3d.visualization.draw_geometries([pcd, mesh_frame])
    
    return points

def nerf_pcd_to_mesh(
        pcd,
        save_name,
        world_bound=[
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0]
        ],
        nerf_scale_factor=3.0,
        translation=0.0,
        alpha=0.03,
        show=False):
    
    pcd = o3d.io.read_point_cloud(pcd) # load nerf point cloud
    # rescaling and translation
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) * nerf_scale_factor + translation)
    # crop to world bound
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=world_bound[0], max_bound=world_bound[1])
    crop_pcd = pcd.crop(bbox)
    # Define the 8 points of the bounding box
    min_bd = world_bound[0]
    max_bd = world_bound[1]
    points = np.array([
        [min_bd[0], min_bd[1], min_bd[2]],
        [max_bd[0], min_bd[1], min_bd[2]],
        [min_bd[0], max_bd[1], min_bd[2]],
        [max_bd[0], max_bd[1], min_bd[2]],
        [min_bd[0], min_bd[1], max_bd[2]],
        [max_bd[0], min_bd[1], max_bd[2]],
        [min_bd[0], max_bd[1], max_bd[2]],
        [max_bd[0], max_bd[1], max_bd[2]],
    ])
    # Define the lines for the box
    indices = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    colors = [[1, 0, 0] for i in range(len(indices))]
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(points)
    lineset.lines = o3d.utility.Vector2iVector(indices)
    lineset.colors = o3d.utility.Vector3dVector(colors)
    
    # show origin and xyz axis
    
    
    if show:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.6, origin=[0.0, 0.0, 0.0])
        # show pcd and the box
        o3d.visualization.draw_geometries([pcd, lineset, mesh_frame])
        # show cropped pcd and the box
        o3d.visualization.draw_geometries([crop_pcd, lineset, mesh_frame])

    # add poind cloud for the base
    grid = np.mgrid[min_bd[0]:max_bd[0]:100j, min_bd[1]:max_bd[1]:100j, 0.0:0.0:1j].reshape(3, -1).T
    grid_point = o3d.utility.Vector3dVector(grid)
    base_pcd = o3d.geometry.PointCloud()
    base_pcd.points = grid_point
    base_pcd.colors = o3d.utility.Vector3dVector(np.ones_like(grid))
    # combine pcd
    merged_pcd = o3d.geometry.PointCloud()
    merged_points = np.concatenate([crop_pcd.points, base_pcd.points], axis=0)
    merged_colors = np.concatenate([np.asarray(crop_pcd.colors), np.asarray(base_pcd.colors)], axis=0)
    merged_pcd.points = o3d.utility.Vector3dVector(merged_points)
    merged_pcd.colors = o3d.utility.Vector3dVector(merged_colors)
    merged_pcd.estimate_normals()

    # filter the point cloud
    # filtered_pcd, ind= merged_pcd.remove_radius_outlier(nb_points=5, radius=0.02)
    filtered_pcd, ind= merged_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
    print(f"Removed {np.asarray(merged_pcd.points).shape[0] - np.asarray(filtered_pcd.points).shape[0]} points.")

    # make water tight mesh
    alpha_shape = alphashape.alphashape(np.asarray(filtered_pcd.points), alpha)
    print(alpha_shape.is_watertight)
    alpha_shape.export(save_name)