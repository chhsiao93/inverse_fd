import numpy as np
from matplotlib import pyplot as plt

# set random seed
np.random.seed(42)
# common info
res = 32
dx = 1 / res
# sand info
cube_size = [1.0, 0.4, 1.0] # x (right), y (up), z (into the screen)
lower_corner = [0.0, 0.0, 0.0]
density = 2 ** 3
# actuator info
plane_center = [0.1, 0.3, 0.5]
plane_size = [0.1, 0.1]
plane_angle = 30.0 # degrees


def cube(lower_corner, cube_size, density):
    cube_vol = cube_size[0] * cube_size[1] * cube_size[2]
    cell_per_cube = cube_vol / (dx ** 3) # number of cells in a cube
    num_particles = int(cell_per_cube * density + 1) # +1 to avoid empty particles
    cube_particles = np.random.rand(num_particles, 3) * cube_size + lower_corner    
    return cube_particles

def surface(x, y):
    return 0.1* (np.sin(1 * np.pi * x) + 0.5*np.sin(1.5 * np.pi * y) + 0.05*np.sin(4 * np.pi * x)) + 0.22

def actuator_plane(plane_center, plane_size, plane_angle):
    # Create a grid of points in the plane
    y = np.linspace(-plane_size[0] / 2, plane_size[0] / 2, 100)
    z = np.linspace(-plane_size[1] / 2, plane_size[1] / 2, 100)
    y, z = np.meshgrid(y, z)
    y = y.flatten()
    z = z.flatten()
    x = np.full_like(y, 0.0)
    
    # Rotate the points around the center
    angle_rad = np.deg2rad(plane_angle)
    # Rotate matrix
    rot = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                    [np.sin(angle_rad), np.cos(angle_rad)]])
    # Rotate along z-axis
    x, y = rot @ np.array([x, y])
    # stack the points
    actuator_particles = np.vstack((x, y, z)).T + plane_center
    
    return actuator_particles

# Generate the cube particles
cube_particles = cube(lower_corner, cube_size, density)
above_surface = cube_particles[:, 1] > surface(cube_particles[:, 0], cube_particles[:, 2])
cube_particles = cube_particles[~above_surface]
print(cube_particles.shape)

# Generate the actuator particles
actuator_particles = actuator_plane(plane_center, plane_size, plane_angle)
print(actuator_particles.shape)

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

# Save the particles to a npz file
np.savez('data/sandbox_particles.npz', sand=cube_particles, actuator=actuator_particles)