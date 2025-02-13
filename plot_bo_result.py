import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import argparse

#input arguments
parser = argparse.ArgumentParser(description='Plot BO result')
parser.add_argument('--var_name', type=str, choices=['phi', 'vel'], default='phi', help='variable name')
parser.add_argument('--loss_name', type=str, choices=['mse', 'ssim', 'depth'], default='mse', help='loss name')

args = parser.parse_args()
var_name = args.var_name
loss_name = args.loss_name
output_name = 'bo_' + var_name
landscape = np.load(f'data/losslandscape_{var_name}.npz', allow_pickle=True)
var_x = np.squeeze(landscape[var_name])
loss = landscape[loss_name]
if loss_name == 'ssim':
    loss = -1.0 * loss
loss = np.mean(loss,axis=1)

interpolate_func = interpolate.interp1d(var_x, loss)
def f(x):
    y = interpolate_func(x[0])
    return y

# reading BO result file
with open(f'result/{output_name}_{loss_name}.pkl', 'rb') as file:
    res = pickle.load(file)

print(f'Opimtized {var_name} is at {res.x}, and the estimate loss is {res.fun}')
n_random_starts = res.specs['args']['n_random_starts']
n_calls = res.specs['args']['n_calls']
space = res.space
x = np.linspace(space.bounds[0][0], space.bounds[0][1], 1000).reshape(-1, 1)

for i in range(n_calls-n_random_starts+1):
    # Extracting the Gaussian process model and the space
    gp = res.models[i]
    gp.fit(res.x_iters[:n_random_starts+i], res.func_vals[:n_random_starts+i])
    # Predicting the mean and standard deviation of the GP at the grid points
    y_pred, sigma = gp.predict(x, return_std=True)
    # Plotting the GP mean and confidence interval
    

    
    if i == n_calls-n_random_starts: # the final plot
        fig, ax = plt.subplots(1,1, figsize=(6, 5))
        # Plot loss landscape
        plt.plot(var_x, loss, 'r-', label='Loss Landscape')
        plt.plot(x, y_pred, 'k-', label='GP mean')
        
        plt.fill_between(x.ravel(), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, alpha=0.2, label='95% confidence interval')
        # # Plot the random initialization
        plt.scatter(res.x_iters[:n_random_starts], res.func_vals[:n_random_starts], marker='^', edgecolor='k', facecolor='none', s=100, label='initial observations')
        
        obs_x = np.array(res.x_iters)[n_random_starts:n_random_starts+i,0]
        obs_y = res.func_vals[n_random_starts:n_random_starts+i]
        # plot the rest of the points with different color
        plt.scatter(obs_x, obs_y, c=np.arange(len(obs_x)), cmap='viridis', s=50, label='BO observation')
        # plot number on the points
        # this won't work for eps file
        for num in range(0, len(obs_x), 2):
            plt.annotate(str(num+1), (obs_x[num], obs_y[num]), xytext=(1, 5), textcoords='offset points')
        
        plt.title('Bayesian Optimization')
        plt.xlabel(var_name)
        plt.ylabel(loss_name)
        plt.xlim(space.bounds[0])
        plt.legend()
        plt.tight_layout()
        # plt.savefig(f'result/bo_{var_name}_{loss_name}.png', dpi=200)
        plt.savefig(f'result/bo_{var_name}_{loss_name}.pdf')
        # plt.savefig(f'result/bo_{var_name}_{loss_name}.eps', format='eps')
        plt.show()