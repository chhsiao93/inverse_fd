import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


data = np.load('data/results.npz', allow_pickle=True)
vel = np.squeeze(data['vel'])
err = np.mean(np.squeeze(data['err']), axis=1)
method = 'BFGS'
opt_data = np.load(f'data/{method}_results.npz', allow_pickle=True)
opt_data_vel = np.squeeze(opt_data['vel'])
opt_data_err = np.mean(np.squeeze(opt_data['err']), axis=1)

fig, ax = plt.subplots()
# for i in range(err.shape[1]):
#     ax.plot(vel, err[:,i], '--', label=f'view {i}')
ax.plot(vel, err,'k-', label='mean')
ax.scatter(opt_data_vel, opt_data_err,s=np.arange(len(opt_data_vel)),facecolor='none',edgecolor='b', label=method)

ax.set_xlabel('Velocity')
ax.set_ylabel('Image looss (Pixel MSE)')
ax.set_yscale('log')
ax.set_title('Loss Landscape')
ax.legend()
plt.tight_layout()
plt.savefig(f'data/{method}_result.png', dpi=200)

fig, ax = plt.subplots()
ax.plot(opt_data_err)
ax.set_xlabel('Iteration')
ax.set_ylabel('Image looss (Pixel MSE)')
ax.set_yscale('log')
plt.tight_layout()
plt.savefig(f'data/{method}_history.png',dpi=200)

# Create a figure and axis
fig, ax = plt.subplots()
# x y limits
xlim = [min(vel.min(), opt_data_vel.min()), max(vel.max(), opt_data_vel.max())]
ylim = [min(err.min(), opt_data_err.min()), max(err.max(), opt_data_err.max())]
# cmap for scatter plot
cmap = plt.get_cmap('viridis')
# Create a scatter plot animation
def update(frame):
    ax.clear()
    ax.plot(vel, err, 'k-', label='Loss Landscape')
    # update scatter plot with edge color based on colormap
    ax.axvline(x=opt_data_vel[frame], color='r', linestyle='--', label='Optimized Velocity')
    ax.scatter(opt_data_vel[:frame], opt_data_err[:frame], c=np.arange(frame), s=np.arange(frame)+1, cmap=cmap, alpha=0.7,  label=f'{method} Results')
    ax.set_xlabel('Velocity')
    ax.set_ylabel('Loss (Pixel MSE)')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_yscale('log')
    ax.set_title('Iteration: %d' % frame)
    ax.legend()
    plt.tight_layout()

animation = FuncAnimation(fig, update, frames=len(opt_data_vel), interval=200)
# Save the animation
animation.save(f'data/{method}_animation.gif', writer='imagemagick', fps=10)