import numpy as np
import open3d as o3d
import mitsuba as mi
import drjit as dr
mi.set_variant('cuda_ad_rgb') # set variant before import anything else
from mitsuba import ScalarTransform4f as T
   
def pcd_to_mesh(pos, mat=None, mask_id=None, save_name='mesh.ply'):
    if mat is None and mask_id is None:
        print('no mask provided')
    else:
        # mask for material
        # print(f'mask material')
        target_type = mask_id # 3: sand, 5: stationary
        pos = pos[mat == target_type].astype('float64') # sand position
    
    # print(pos.shape)
    # print(np.mean(pos,axis=0))
    # Create a point cloud for sand
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pos)
    # print("point cloud created")
    alpha = 0.03
    # print(f"alpha={alpha:.3f}")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.8, 0.7, 0.6])
    # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    o3d.io.write_triangle_mesh(f"{save_name}", mesh)
    # print(f'mesh is saved at {save_name}.')
    
    return save_name

def multi_view(ply_file, n_view=2, from_o3d=False, save_prefix=''):
    sand_color = [0.8, 0.7, 0.6]
    # Define the scene to render
    if from_o3d:
        mesh = o3d.io.read_triangle_mesh(ply_file)
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        mesh_t.material.set_default_properties()
        mesh_t.material.material_name = 'defaultLit'
        mi_mesh = mesh_t.to_mitsuba('sand')
        # Define the scene to render
        scene_dict = {
            'type': 'scene',
            'integrator': {'type': 'path'},
            'light': {
                'type': 'point',
                'position': [5.0, 5.0, 0.0],
                'intensity': {
                    'type': 'spectrum',
                    'value': 20.0,
                }
            },
            'sand':mi_mesh
        }
    else:
        scene_dict = {
            'type': 'scene',
            'integrator': {'type': 'path'},
            'light': {
                'type': 'point',
                'position': [5.0, 5.0, 0.0],
                'intensity': {
                    'type': 'spectrum',
                    'value': 20.0,
                }
            },
            'sand':{
            'type': 'ply',
            'filename': ply_file,
            'bsdf': {
                'type' : 'diffuse',
                'reflectance': {'type': 'rgb', 'value': sand_color},
                }
            }
        }
    scene = mi.load_dict(scene_dict)
    print('loaded the scene')
    
    sensor_count = n_view

    radius = 2.5
    phis = [30.0 * i for i in range(sensor_count)]
    theta = 60.0

    sensors = [load_sensor(radius, phi, theta) for phi in phis]
    images = [mi.render(scene, spp=16, sensor=sensor) for sensor in sensors]
    return images, phis

def load_sensor(r, phi, theta):
    # Apply two rotations to convert from spherical coordinates to world 3D coordinates.
    origin = T.rotate([0, 1, 0], phi).rotate([1, 0, 0], theta) @ mi.ScalarPoint3f([0, r, 0])
    return mi.load_dict({
        'type': 'perspective',
        'fov': 39.3077,
        'to_world': T.look_at(
            origin=origin, 
            target=[0.5, 0, 0.5], 
            up=[0, 1, 0]
        ),
        'sampler': {
            'type': 'independent', 
            'sample_count': 16
        },
        'film': {
            'type': 'hdrfilm',
            'width': 256,
            'height': 256,
            'rfilter': {
                'type': 'tent',
            },
            'pixel_format': 'rgb',
        },
    })
def compute_image_loss(image, image_ref):
    return (np.mean((image - image_ref)**2))
    