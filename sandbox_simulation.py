import taichi as ti
import numpy as np
from engine.mpm_solver import MPMSolver
import os

def forward(phi_degree=45.0, sand_density=1.0, sand_E_scale=1.0, npz_path=None, save_fname=None, save_frame=None):
    
    ti.init(arch=ti.cuda, device_memory_GB=15)  # Try to run on GPU

    gui = ti.GUI("Taichi Elements", res=512, background_color=0x112F41, show_gui=False)

    mpm = MPMSolver(res=(32, 32, 32), phi_degree=phi_degree, sand_density=sand_density, sand_E_scale=sand_E_scale)
    mpm.set_gravity([0.0, -9.81, 0.0])
    frame_dt = 0.02
    pos = [] # store position data
    stress = [] # store stress data
    # load the actuator particles
    act_npz_path = f'data/actuator_particles.npz'
    actuator_particles = np.load(act_npz_path, allow_pickle=True)['actuator']
    
    if npz_path is None:
        # get the true initial state of sand particles (from MPM)
        sand_npz_path = f'data/sand_particles.npz'
        sand_particles =  np.load(sand_npz_path, allow_pickle=True)['sand']
    else:
        data = np.load(npz_path, allow_pickle=True)
        sand_particles = data['sand']

    # adding a actuactor particles
    mpm.add_particles(particles=actuator_particles,
                    material=mpm.material_actuator,
                    color=0xFFFF99)
    # add the sand particles
    mpm.add_particles(particles=sand_particles,
                    material=mpm.material_sand,
                    )
    # print("Lower the plaw")
    for frame in range(25):
        mpm.step(frame_dt=frame_dt, act_v=np.array([0.0,-0.1,0.0], dtype=np.float32))
        particles = mpm.particle_info() 
        pos.append(np.array(particles['position']))
        stress.append(np.array(particles['stress']))
        
    # print("Push the plaw")
    for frame in range(75):
        mpm.step(frame_dt=frame_dt, act_v=np.array([0.3,0.0,0.0], dtype=np.float32))
        particles = mpm.particle_info() 
        pos.append(np.array(particles['position']))
        stress.append(np.array(particles['stress']))
        
    # print("Pause the plaw")
    for frame in range(25):
        mpm.step(frame_dt=frame_dt, act_v=np.array([0.0,0.0,0.0], dtype=np.float32))
        particles = mpm.particle_info() 
        pos.append(np.array(particles['position']))
        stress.append(np.array(particles['stress']))
    
    # print("Withdraw the plaw")
    for frame in range(75):
        mpm.step(frame_dt=frame_dt, act_v=np.array([-0.3,0.0,0.0], dtype=np.float32))
        particles = mpm.particle_info() 
        pos.append(np.array(particles['position']))
        stress.append(np.array(particles['stress']))
        
    mat = np.array(particles['material'])
    pos = np.array(pos)
    stress = np.array(stress)[:, mat==5, :, :] # only store the stress for actuactor
    # print(pos.shape, mat.shape, stress.shape)
    
    
    if save_fname is not None:
        if save_frame is None:
            np.savez(save_fname, pos=pos, mat=mat, stress=stress)
        else:
            # save only the specified frame
            # save_frame is a list of frame indices, e.g., [0, 10, 20]
            np.savez(save_fname, pos=pos[save_frame], mat=mat, stress=stress[save_frame])
    
    return pos, mat, stress