import numpy as np
import matplotlib.pyplot as plt


data = np.load('data/results.npz', allow_pickle=True)
vel = np.squeeze(data['vel'])
err = np.squeeze(data['err'])

assert vel.shape[0] == err.shape[0]
print('n_view:', err.shape[1])
print(err[:,0], err[:1])
fig, ax = plt.subplots()
for i in range(err.shape[1]):
    ax.plot(vel, err[:,i], '--', label=f'view {i}')
ax.plot(vel, np.mean(err,axis=1),'k-', label='mean')
ax.set_xlabel('Velocity')
ax.set_ylabel('Image looss (Pixel MSE)')
ax.set_yscale('log')
ax.set_title('Loss Landscape')
ax.legend()
plt.tight_layout()
plt.savefig('data/losslandscape_vel.png', dpi=200)