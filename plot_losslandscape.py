import numpy as np
import matplotlib.pyplot as plt

# Assuming var_name is defined somewhere above this code
var_name = 'phi'

xlabel = 'actuator velocity'
ground_truth = -0.2
xmin, xmax = -0.3, 0.1
if var_name == 'phi':
    xlabel = 'friction angle (degree)'
    ground_truth = 45.0
    xmin, xmax = 35.0, 50.0

data = np.load(f'data/losslandscape_{var_name}.npz', allow_pickle=True)
x = np.squeeze(data[var_name]) # vel or phi

colors = plt.get_cmap('tab20', 12)  # Get a colormap with 12 distinct colors

for loss_name in ["mse", "ssim", "depth"]:
    loss = data[loss_name]
    n_view = loss.shape[1]
    if loss_name == 'ssim':
        loss = -1.0 * loss
    mean_loss = np.mean(loss, axis=1)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(x, mean_loss, 'k-', lw=2, label='mean')
    ax.scatter(x[np.argmin(mean_loss)], np.min(mean_loss), c='k', s=30, alpha=0.8)  # best view
    for view in range(n_view):
        color = colors(view % 12)  # Cycle through the 12 colors
        ax.plot(x, loss[:, view], '--', label=f'view {view}', color=color)  # plot each view
        ax.scatter(x[np.argmin(loss[:, view])], np.min(loss[:, view]), color=color, s=30, alpha=0.8)  # best view

    ax.axvline(ground_truth, c='r', lw=2, label='ground truth')
    ax.set_xlabel(xlabel, fontsize=15)  # actuator velocity or friction angle (degree)
    ax.set_ylabel(f'Loss ({loss_name})', fontsize=15)  # MSE or SSIM or Depth
    ax.set_xlim(xmin, xmax)
    # set up ymin for each loss
    if loss_name == "mse":
        ax.set_ylim(13e-5, 38e-5)
    elif loss_name == "ssim":
        ax.set_ylim(-0.73, -0.64)
    elif loss_name == "depth":
        ax.set_ylim(0.018, 0.05)

    if loss_name == "mse":
        # 3 cols of legends
        ax.legend(loc='upper right', fontsize=10, ncol=4, columnspacing=0.5)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'result/losslandscape_{var_name}_{loss_name}.eps', format='eps', dpi=300)  # save as eps