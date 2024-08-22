
from main import PINN, x_min, x_max, y_min, y_max
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pinn = PINN()
pinn.net.load_state_dict(torch.load("adapt1000.pt"))

x = np.arange(x_min, x_max, 0.001)
y = np.arange(y_min, y_max, 0.001)
X, Y = np.meshgrid(x, y)
x = X.reshape(-1, 1)
y = Y.reshape(-1, 1)

dst_from_cyl = np.sqrt((x - 0) ** 2 + (y - 0) ** 2)
# cyl_mask = dst_from_cyl > r
cyl_mask = dst_from_cyl >0

xy = np.concatenate([x, y], axis=1)
xy = torch.tensor(xy, dtype=torch.float32).to(device)

# with torch.no_grad():
#     u, v, p = pinn.predict(xy)
#     u = u.cpu().numpy()
#     u = np.where(cyl_mask, u, np.nan).reshape(Y.shape)
#     v = v.cpu().numpy()
#     v = np.where(cyl_mask, v, np.nan).reshape(Y.shape)
#     p = p.cpu().numpy()
#     p = np.where(cyl_mask, p, np.nan).reshape(Y.shape)
#     V = np.sqrt(u ** 2 + v ** 2)

u, v, p = pinn.predict(xy)
with torch.no_grad():
    u = u.cpu().numpy()
    u = np.where(cyl_mask, u, np.nan).reshape(Y.shape)
    v = v.cpu().numpy()
    v = np.where(cyl_mask, v, np.nan).reshape(Y.shape)
    p = p.cpu().numpy()
    p = np.where(cyl_mask, p, np.nan).reshape(Y.shape)
    V = np.sqrt(u ** 2 + v ** 2)

fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)

data = (u, v, p,V)
labels = ["$u(x,y)$", "$v(x,y)$", "$p(x,y)$","$V(x,y)$"]
for i in range(4):
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
fig.tight_layout()
fig.savefig("readapt_1000", dpi=500)




