import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

prefix = ''
var_name = 'phi'
err_name = 'mse'
# landscape data
landscape = np.load(f'data/losslandscape_{var_name}.npz', allow_pickle=True)
var_x = np.squeeze(landscape[var_name])
err = np.mean(np.squeeze(landscape[err_name]), axis=1)
# optimized result
method = 'Nelder-Mead'
opt_data = np.load(f'data/{var_name}_{method}_results.npz', allow_pickle=True)
opt_var_x = np.squeeze(opt_data['guess'])
opt_err = np.mean(np.squeeze(opt_data['err']), axis=1)

fig, ax = plt.subplots()
# for i in range(err.shape[1]):
#     ax.plot(var_x, err[:,i], '--', label=f'view {i}')
ax.plot(var_x, err,'k-', label='mean')
ax.scatter(opt_var_x, opt_err,s=np.arange(len(opt_var_x)),facecolor='none',edgecolor='b', label=method)

ax.set_xlabel(var_name)
ax.set_ylabel(f'Loss ({err_name})')
ax.set_yscale('log')
ax.set_title('Loss Landscape')
ax.legend()
plt.tight_layout()
plt.savefig(f'data/{prefix}{var_name}_{method}_result.png', dpi=200)

fig, ax = plt.subplots()
ax.plot(opt_err)
ax.set_xlabel('Iteration')
ax.set_ylabel(f'Loss ({err_name})')
ax.set_yscale('log')
plt.tight_layout()
plt.savefig(f'data/{prefix}{var_name}_{method}_history.png',dpi=200)

# Create a figure and axis
fig, ax = plt.subplots()
# x y limits
xlim = [min(var_x.min(), opt_var_x.min()), max(var_x.max(), opt_var_x.max())]
ylim = [min(err.min(), opt_err.min()), max(err.max(), opt_err.max())]
# cmap for scatter plot
cmap = plt.get_cmap('viridis')
# Create a scatter plot animation
def update(frame):
    ax.clear()
    ax.plot(var_x, err, 'k-', label='Loss Landscape')
    # update scatter plot with edge color based on colormap
    ax.axvline(x=opt_var_x[frame], color='r', linestyle='--', label='Optimized var_xocity')
    ax.scatter(opt_var_x[:frame], opt_err[:frame], c=np.arange(frame), s=np.arange(frame)+1, cmap=cmap, alpha=0.7,  label=f'{method} Results')
    ax.set_xlabel(var_name)
    ax.set_ylabel(f'Loss ({err_name})')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_yscale('log')
    ax.set_title('Iteration: %d' % frame)
    ax.legend()
    plt.tight_layout()

animation = FuncAnimation(fig, update, frames=len(opt_var_x), interval=200)
# Save the animation
animation.save(f'data/{prefix}{var_name}_{method}_animation.gif', writer='imagemagick', fps=10)