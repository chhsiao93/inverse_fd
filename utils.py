import numpy as np
import open3d as o3d
from skimage.metrics import structural_similarity as ssim
   
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
    