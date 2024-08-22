
from cylinder_continue import PINN, x_min, x_max, y_min, y_max,r,xc,yc
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pinn = PINN()
pinn.net.load_state_dict(torch.load("weight.pt"))

x = np.arange(x_min, x_max, 0.001)
y = np.arange(y_min, y_max, 0.001)
X, Y = np.meshgrid(x, y)
x = X.reshape(-1, 1)
y = Y.reshape(-1, 1)

dst_from_cyl = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
# cyl_mask = dst_from_cyl > r
cyl_mask = dst_from_cyl >r

t=np.zeros((x.shape[0],1))
xyt = np.concatenate([x, y,t], axis=1)
xyt[:,2]=10

for k in range(10):
    xyt[:, 2] = k
    xyt = torch.tensor(xyt, dtype=torch.float32).to(device)

    with torch.no_grad():
        u, v, p = pinn.predict(xyt)
        u = u.cpu().numpy()
        u = np.where(cyl_mask, u, np.nan).reshape(Y.shape)
        v = v.cpu().numpy()
        v = np.where(cyl_mask, v, np.nan).reshape(Y.shape)
        p = p.cpu().numpy()
        p = np.where(cyl_mask, p, np.nan).reshape(Y.shape)

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)

    data = (u, v, p)
    labels = ["$u(x,y)$", "$v(x,y)$", "$p(x,y)$"]
    for i in range(3):
        ax = axes[i]
        im = ax.imshow(
            data[i], cmap="rainbow", extent=[x_min, x_max, y_min, y_max], origin="lower"
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad="3%")
        fig.colorbar(im, cax=cax, label=labels[i])
        ax.set_title(labels[i])
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_aspect("equal")

    file_name = f"plot_{k}.png"
    fig.tight_layout()
    fig.savefig(file_name, dpi=500)
    plt.close(fig)  # 关闭图像，释放内存




