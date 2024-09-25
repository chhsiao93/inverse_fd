import numpy as np
import matplotlib.pyplot as plt

var_name = 'vel'
xlabel = 'actuator velocity'
ground_truth = -0.2
xmin, xmax = -0.3, 0.1
data = np.load(f'data/losslandscape_{var_name}.npz', allow_pickle=True)
# data = np.load('data/losslandscape_vel.npz', allow_pickle=True)
x = np.squeeze(data[var_name]) #vel or phi
mse = data['mse']
ssim = -1.0*data['ssim'] # negative ssim
depth = data['depth']
mean_mse = np.mean(mse,axis=1)
mean_ssim = np.mean(ssim,axis=1)
mean_depth = np.mean(depth,axis=1)
n_view = mse.shape[1]
print(f'n view: {n_view}')
fig, ax = plt.subplots(1,3,figsize=(18,6))
ax[0].plot(x, mean_mse, 'k-', label='mean', lw=2)
ax[1].plot(x, mean_ssim, 'k-', label='mean', lw=2)
ax[2].plot(x, mean_depth, 'k-', label='mean', lw=2)
for view in range(n_view):
    ax[0].plot(x, mse[:,view], '--', label=f'view {view}')
    ax[1].plot(x, ssim[:,view], '--', label=f'view {view}')
    ax[2].plot(x, depth[:,view], '--', label=f'view {view}')
ax[0].set_xlabel(xlabel, fontsize=15) #actuator velocity or friction angle (degree)
ax[1].set_xlabel(xlabel, fontsize=15) 
ax[2].set_xlabel(xlabel, fontsize=15) 
ax[0].set_ylabel('MSE (RGB)', fontsize=15)
ax[1].set_ylabel('Negative SSIM', fontsize=15)
ax[2].set_ylabel('MSE (Depth)', fontsize=15)
ax[0].set_xlim(xmin, xmax)
ax[1].set_xlim(xmin, xmax)
ax[2].set_xlim(xmin, xmax)
ax[0].legend()
ax[1].legend()
ax[2].legend()
ax[0].axvline(ground_truth, c='r', lw=2)
ax[1].axvline(ground_truth, c='r', lw=2)
ax[2].axvline(ground_truth, c='r', lw=2)
plt.tight_layout()
plt.savefig(f'{var_name}_losslandscape.png', dpi=200)