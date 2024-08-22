
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.autograd import grad
from pyDOE import lhs
import matplotlib.pyplot as plt

from network import DNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_min = 0.0
x_max = 1.0
y_min = 0.0
y_max = 1.0



ub = np.array([x_max, y_max])
lb = np.array([x_min, y_min])


N_w = 2500  # wall

def getData():
    left_xy=np.random.uniform([x_min, y_min], [x_min, y_max], (N_w, 2))
    left_uv=np.zeros((N_w, 2))


    right_xy = np.random.uniform([x_max, y_min], [x_max, y_max], (N_w, 2))
    right_uv = np.zeros((N_w, 2))

    bottom_xy = np.random.uniform([x_min, y_min], [x_max, y_min], (N_w, 2))
    bottom_uv = np.zeros((N_w, 2))


    up_xy = np.random.uniform([x_min, y_max], [x_max, y_max], (N_w, 2))
    up_u = np.ones((N_w, 1))

    up_v = np.zeros((N_w, 1))
    up_uv=np.concatenate([up_u,up_v], axis=1)

    bund_xy=np.concatenate([up_xy,left_xy, right_xy, bottom_xy], axis=0)
    bund_uv = np.concatenate([up_uv,left_uv, right_uv, bottom_uv], axis=0)

    in_xy=np.random.uniform([x_min, y_min], [x_max, y_max], (4*N_w, 2))


    bund_xy= torch.tensor(bund_xy, dtype=torch.float32).to(device)
    bund_uv = torch.tensor(bund_uv, dtype=torch.float32).to(device)
    in_xy = torch.tensor(in_xy, dtype=torch.float32).to(device)

    return bund_xy,bund_uv,in_xy

bund_xy,bund_uv,in_xy=getData()

def get_govern_data():
    # 读取CSV文件
    df = pd.read_csv('re1000uvp.csv')
    print(df.head())


    # 划分特征和目标变量
    X = df[['x', 'y']].values
    y = df[['u', 'v','sqrt']].values
    print("X_train:\n")
    # 划分为训练集和测试集，80% 作为训练集，20% 作为测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1,test_size=0.9, random_state=42)
    data_xy = torch.tensor(X_train, dtype=torch.float32).to(device)
    data_uv = torch.tensor(y_train, dtype=torch.float32).to(device)
    return data_xy,data_uv


data_xy,data_uv=get_govern_data()



class PINN:


    def __init__(self) -> None:
        self.net = DNN(dim_in=2, dim_out=3, n_layer=4, n_node=40, ub=ub, lb=lb).to(
            device
        )
        self.lbfgs = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1,
            max_iter=50000,
            max_eval=50000,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            history_size=50,
            line_search_fn="strong_wolfe",
        )
        self.adam = torch.optim.Adam(self.net.parameters(), lr=5e-4)
        self.losses = {"bc": [],  "pde": [], "data": []}
        self.iter = 0
        self.re = 0

    def predict(self, xy):
        out = self.net(xy)
        u = out[:, 0:1]
        v = out[:, 1:2]
        p = out[:, 2:3]

        return u, v, p


    def bc_loss(self, xy):
        u, v = self.predict(xy)[0:2]

        mse_bc = torch.mean(torch.square(u - bund_uv[:, 0:1])) + torch.mean(
            torch.square(v - bund_uv[:, 1:2])
        )

        return mse_bc

    def data_loss(self, xy):
        u, v = self.predict(xy)[0:2]

        mse_data = torch.mean(torch.square(u - data_uv[:, 0:1])) + torch.mean(
            torch.square(v - data_uv[:, 1:2])
        )

        return mse_data


    def pde_loss(self, xy):
        xy = xy.clone()
        xy.requires_grad = True
        u, v, p = self.predict(xy)

        u_out = grad(u.sum(), xy, create_graph=True)[0]
        v_out = grad(v.sum(), xy, create_graph=True)[0]

        p_out = grad(p.sum(), xy, create_graph=True)[0]
        p_x = p_out[:, 0:1]
        p_y = p_out[:, 1:2]


        u_x = u_out[:, 0:1]
        u_y = u_out[:, 1:2]
        v_x = v_out[:, 0:1]
        v_y = v_out[:, 1:2]

        # 在你的 pde_loss 方法中的代码中继续
        u_xx = grad(u_x.sum(), xy, create_graph=True)[0][:, 0:1]
        u_yy = grad(u_y.sum(), xy, create_graph=True)[0][:, 1:2]

        v_xx = grad(v_x.sum(), xy, create_graph=True)[0][:, 0:1]
        v_yy = grad(v_y.sum(), xy, create_graph=True)[0][:, 1:2]



        # continuity equation
        f0 = u_x + v_y

        f1 = u * u_x + v * u_y+p_x -(u_xx +u_yy)/self.re
        f2 = u * v_x + v * v_y+p_y - (v_xx +v_yy)/self.re



        mse_f0 = torch.mean(torch.square(f0))
        mse_f1 = torch.mean(torch.square(f1))
        mse_f2 = torch.mean(torch.square(f2))

        mse_pde = mse_f0 + mse_f1 + mse_f2

        return mse_pde

    def closure(self):
        self.lbfgs.zero_grad()
        self.adam.zero_grad()

        mse_bc = self.bc_loss(bund_xy)
        mse_pde = self.pde_loss(in_xy)
        mse_data = self.data_loss(data_xy)

        # if self.iter%100==0:
        #     grad_loss_pde = torch.autograd.grad(outputs=mse_pde, inputs=self.net.parameters(), retain_graph=True,
        #                                         create_graph=True)
        #     grad_loss_pde_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grad_loss_pde))
        #
        #     grad_loss_bc = torch.autograd.grad(outputs=mse_bc, inputs=self.net.parameters(), retain_graph=True,
        #                                        create_graph=True)
        #     grad_loss_bc_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grad_loss_bc))
        #
        #     # 平衡权重
        #     lambda_pde = (grad_loss_pde_norm + grad_loss_bc_norm) / grad_loss_pde_norm
        #     lambda_bc = (grad_loss_pde_norm + grad_loss_bc_norm) / grad_loss_bc_norm
        #
        #     # 总损失
        #     loss = lambda_pde * mse_pde + lambda_bc * mse_bc
        #
        # else:
        #     loss = mse_bc + mse_pde
        loss = mse_bc + mse_pde+mse_data
        loss.backward()

        self.losses["bc"].append(mse_bc.detach().cpu().item())
        # self.losses["outlet"].append(mse_outlet.detach().cpu().item())
        self.losses["pde"].append(mse_pde.detach().cpu().item())
        self.losses["data"].append(mse_data.detach().cpu().item())
        self.iter += 1

        print(
            f"\r{self.iter} Loss: {loss.item():.5e} BC: {mse_bc.item():.3e}  pde: {mse_pde.item():.3e} pde: {mse_data.item():.3e}",
            end="",
        )

        if self.iter % 500 == 0:
            print("")
        return loss

def plotLoss(losses_dict, path, info=[ "BC", "PDE","DATA"]):
    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(10, 6))
    axes[0].set_yscale("log")
    for i, j in zip(range(3), info):
        axes[i].plot(losses_dict[j.lower()])
        axes[i].set_title(j)
    plt.show()
    fig.savefig(path)



if __name__ == "__main__":
    pinn = PINN()

    pinn.re=1000
    for i in range(20000):
        pinn.closure()
        pinn.adam.step()
    pinn.lbfgs.step(pinn.closure)
    torch.save(pinn.net.state_dict(), "re1000.pt")

    plotLoss(pinn.losses, "loss_curve_re600_1.png", ["BC", "PDE","DATA"])




