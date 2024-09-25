import taichi as ti
import numpy as np
from engine.mpm_solver import MPMSolver
import os

def forward(act_vel, f_a=0.0, f_b=0.0, f_c=0.0, force_control=False, phi_degree=45.0, traj=False):
    positions = []
    materials = []
    velocities = []
    original_velocities = []
    npz_path = f'data/nerf_to_mtpts_frame0.npz'

    data = np.load(npz_path, allow_pickle=True)
    key = list(data.keys())[0]  # Gets the first key in the .npz file
    particles = data[key]
    # print(particles.shape)

    ti.init(arch=ti.cuda, device_memory_GB=15)  # Try to run on GPU

    gui = ti.GUI("Taichi Elements", res=512, background_color=0x112F41, show_gui=False)

    mpm = MPMSolver(res=(32, 32, 32), act_vel=act_vel, f_a=f_a, f_b=f_b, f_c=f_c, force_control=force_control, phi_degree=phi_degree)
    mpm.set_gravity([0, -9.81, 0])

    # adding a slope plane
    slope_a = 60
    slope_l = 0.1
    lower_corner = [0.8,0.28,0.45]
    x_ = np.linspace(0.0,slope_l,100)
    z_ = np.linspace(0.0,slope_l,100)
    x, z = np.meshgrid(x_,z_)
    y = x*np.tan(np.radians(slope_a))
    slope_pos = np.vstack((x.reshape(-1),y.reshape(-1),z.reshape(-1))).swapaxes(0,1)
    slope_pos += lower_corner
    # print(slope_pos.shape)
    mpm.add_particles(particles=slope_pos,
                    material=mpm.material_actuator,
                    color=0xFFFF99)

    mpm.add_particles(particles=particles,
                    material=mpm.material_sand,
                    )
    for frame in range(200):
        mpm.step(8e-3)
        if traj:
            particles = mpm.particle_info() # get the last step info 
            positions.append(particles['position'])
            velocities.append(particles['velocity'])
            original_velocities.append(particles['original_velocity'])
    particles = mpm.particle_info() # get the last step info 
    mat = np.array(particles['material'])
    
    if traj:
        return np.array(positions), mat, np.array(velocities), np.array(original_velocities)
    else:
        pos = np.array(particles['position'])
        return pos, mat