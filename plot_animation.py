import numpy as np
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

# loss landscape
data = np.load('data/results.npz', allow_pickle=True)
vel = np.squeeze(data['vel'])
err = np.mean(np.squeeze(data['err']), axis=1)
# BFGS results
method = 'Nelder-Mead'
bfgs = np.load(f'data/{method}_results.npz', allow_pickle=True)
bfgs_vel = np.squeeze(bfgs['vel'])
bfgs_err = np.mean(np.squeeze(bfgs['err']), axis=1)
# Create a figure and axis
fig, ax = plt.subplots()
# x y limits
xlim = [min(vel.min(), bfgs_vel.min()), max(vel.max(), bfgs_vel.max())]
ylim = [min(err.min(), bfgs_err.min()), max(err.max(), bfgs_err.max())]
# cmap for scatter plot
cmap = plt.get_cmap('viridis')
# Create a scatter plot animation
def update(frame):
    ax.clear()
    ax.plot(vel, err, 'k-', label='Loss Landscape')
    # update scatter plot with edge color based on colormap
    ax.axvline(x=bfgs_vel[frame], color='r', linestyle='--', label='Optimized Velocity')
    ax.scatter(bfgs_vel[:frame], bfgs_err[:frame], c=np.arange(frame), s=np.arange(frame)+1, cmap=cmap, alpha=0.7,  label=f'{method} Results')
    ax.set_xlabel('Velocity')
    ax.set_ylabel('Loss (Pixel MSE)')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_yscale('log')
    ax.set_title('Iteration: %d' % frame)
    ax.legend()
    plt.tight_layout()

animation = FuncAnimation(fig, update, frames=len(bfgs_vel), interval=200)
# Save the animation
animation.save(f'data/{method}_animation.gif', writer='imagemagick', fps=10)