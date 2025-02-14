import taichi as ti
import numpy as np
from engine.mpm_solver import MPMSolver



def forward():
    
    ti.init(arch=ti.cuda, device_memory_GB=15)  # Try to run on GPU

    gui = ti.GUI("Taichi Elements", res=512, background_color=0x112F41, show_gui=False)

    mpm = MPMSolver(res=(32, 32, 32))
    mpm.set_gravity([0, -9.81, 0])

    # load the particles
    npz_path = f'data/sandbox_particles.npz'
    data = np.load(npz_path, allow_pickle=True)
    sand_particles = data['sand']
    actuator_particles = data['actuator']

    # adding a actuactor particles
    mpm.add_particles(particles=actuator_particles,
                    material=mpm.material_actuator,
                    color=0xFFFF99)
    # add the sand particles
    mpm.add_particles(particles=sand_particles,
                    material=mpm.material_sand,
                    )
    for frame in range(10):
        print(f"frame: {frame}")
        mpm.step(frame_dt=8e-3, act_v=np.array([0.2,0.0,0.0], dtype=np.float32))
        # mpm.step(8e-3)
        
    
    particles = mpm.particle_info() # get the last step info 
    mat = np.array(particles['material'])
    pos = np.array(particles['position'])
    
    return pos, mat

pos, mat = forward()
cube_particles = pos[mat == 3]
actuator_particles = pos[mat == 5]
# visualize the particles
import matplotlib.pyplot as plt
# Visualize the particles
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(cube_particles[:, 0], cube_particles[:, 2], cube_particles[:, 1], c=cube_particles[:, 1], cmap='coolwarm', s=0.5, alpha=0.1)
ax.scatter(actuator_particles[:, 0], actuator_particles[:, 2], actuator_particles[:, 1], c='k', s=1)
# set x, y, z limits to (0,1)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# perspective angle
ax.view_init(elev=45, azim=-105)
plt.show()
