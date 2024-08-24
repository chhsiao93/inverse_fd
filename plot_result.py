import numpy as np
import matplotlib.pyplot as plt


data = np.load('data/results.npz', allow_pickle=True)
vel = np.squeeze(data['vel'])
err = np.squeeze(data['err'])
bfgs = np.load('data/bfgs_results.npz', allow_pickle=True)
bfgs_vel = np.squeeze(bfgs['vel'])
bfgs_err = np.squeeze(bfgs['err'])
assert vel.shape[0] == err.shape[0]
print('n_view:', err.shape[1])
print(err[:,0], err[:1])
fig, ax = plt.subplots()
# for i in range(err.shape[1]):
#     ax.plot(vel, err[:,i], '--', label=f'view {i}')
ax.plot(vel, np.mean(err,axis=1),'k-', label='mean')
ax.scatter(bfgs_vel, np.mean(bfgs_err,axis=1),s=np.arange(len(bfgs_vel)),facecolor='none',edgecolor='b', label='BFGS')

ax.set_xlabel('Velocity')
ax.set_ylabel('Image looss (Pixel MSE)')
ax.set_yscale('log')
ax.set_title('Loss Landscape')
ax.legend()
plt.tight_layout()
plt.savefig('data/bfgs_result.png', dpi=200)

fig, ax = plt.subplots()
ax.plot(np.mean(bfgs_err,axis=1))
ax.set_xlabel('Iteration')
ax.set_ylabel('Image looss (Pixel MSE)')
ax.set_yscale('log')
plt.tight_layout()
plt.savefig('data/bfgs_history.png',dpi=200)