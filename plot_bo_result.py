import imageio.v2 as imageio
import pickle
import matplotlib.pyplot as plt
import numpy as np
from skopt.plots import plot_convergence, plot_gaussian_process
from scipy import interpolate

var_name = 'phi'
loss_name = 'mse'
acq_func = 'EI'
output_name = 'test'
var_bound = [(35.0, 55.0)]
n_random_starts = 5 # initialize Gaussian prior with 5 random evaluations
n_calls = 15 # total evaluation in Bayesian, including random start.

# get true loss landscape
loss_landscape = np.load(f'data/losslandscape_{var_name}.npz', allow_pickle=True)
loss_x = loss_landscape[var_name]
loss_y = loss_landscape[loss_name]
if len(loss_y.shape) >1:
    loss_y = np.mean(loss_y, axis=-1)
assert loss_x.shape[0] == loss_y.shape[0]
print(loss_x.shape, loss_y.shape)

# use loss landscape as true objective funciton
interpolate_func = interpolate.interp1d(loss_x, loss_y)
def f(x):
    y = interpolate_func(x[0])
    return y

# reading BO result file
with open(f'result/{output_name}.pkl', 'rb') as file:
    res = pickle.load(file)
    
print(f'Opimtized {var_name} is at {res.x}, and the estimate loss is {res.fun}')
# plot
ax = plot_convergence(res)
plt.savefig(f'result/convergence_{var_name}_{loss_name}.png',dpi=200)

for frame in range(n_calls-n_random_starts+1):
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    # Update the data for the line plot
    plot_gaussian_process(res, n_calls=frame,
                               objective=f,
                               noise_level=0,
                               show_legend=True, show_title=False,
                               show_next_point=True, show_acq_func=True, ax=ax[0])
    plot_gaussian_process(res, n_calls=frame,
                               show_legend=True, show_title=False,
                               show_mu=False, show_acq_func=True,
                               show_observations=False,
                               show_next_point=True, ax=ax[1])
    ax[0].set_title(f'Iteration: {frame}')
    ax[0].set_xlabel(var_name)
    ax[0].set_ylabel('Loss')
    ax[0].set_xlim(var_bound[0][0],var_bound[0][1])
    ax[1].set_xlabel(var_name)
    ax[1].set_ylabel(acq_func)
    ax[1].set_xlim(var_bound[0][0],var_bound[0][1])
    plt.tight_layout()
    plt.savefig(f'result/bo_{var_name}_{frame:02d}.png',dpi=200)
    
# make png to gif
images = []
for frame in range(n_calls-n_random_starts+1):
    images.append(imageio.imread(f'result/bo_{var_name}_{frame:02d}.png'))
imageio.mimsave(f'result/bo_{var_name}_{loss_name}.gif', images, format='GIF', fps=2)
