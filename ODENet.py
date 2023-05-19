import os
import argparse
import time
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=100)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=10000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

sampling_rate = 5e3
sampling_rate_str = '5e3'
fading_type = 'rician'

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
# t = torch.linspace(1. / sampling_rate, 100. / sampling_rate, args.data_size).to(device)
indices = sorted(np.random.choice(list(range(101)), 101, replace=False))
# indices = list(range(0, 101, 1))
t = torch.tensor([i / sampling_rate for i in indices[1:]]).to(device)

with open('eigentriples_' + sampling_rate_str + '.pkl', 'rb') as file:
    rayleigh_gain = pickle.load(file)
    rayleigh_w = pickle.load(file)
    rayleigh_lv = pickle.load(file)
    rayleigh_rv = pickle.load(file)
    rician_gain = pickle.load(file)
    rician_w = pickle.load(file)
    rician_lv = pickle.load(file)
    rician_rv = pickle.load(file)
    if fading_type == 'rayleigh':
        gain, w, lv, rv = rayleigh_gain, rayleigh_w, rayleigh_lv, rayleigh_rv
    if fading_type == 'rician':
        gain, w, lv, rv = rician_gain, rician_w, rician_lv, rician_rv

m = np.median(w)

# true_y = [[w[1] / m] + list(lv[1]) + list(rv[1]) + list(gain[1].ravel() / m) + list((gain[0].ravel()) / m)]
# for i in range(1, 100):
    # true_y.append([w[i + 1] / m] + list(lv[i + 1]) + list(rv[i + 1]) + list(gain[i + 1].ravel() / m) + list(gain[i].ravel() / m))
# true_y0 = true_y[0]
# true_y0 = torch.tensor([true_y0], dtype=torch.float32).to(device)
# true_y = torch.tensor([[y] for y in true_y], dtype=torch.float32).to(device)
true_y = [[w[indices[1]] / m] + list(lv[indices[1]]) + list(rv[indices[1]]) + list(gain[indices[1]].ravel() / m) + list((gain[indices[0]].ravel()) / m) + [indices[1] - indices[0]]]
for i in range(1, 100):
    true_y.append([w[indices[i + 1]] / m] + list(lv[indices[i + 1]]) + list(rv[indices[i + 1]]) + list(gain[indices[i + 1]].ravel() / m) + list(gain[indices[i]].ravel() / m) + [indices[i + 1] - indices[i]])
true_y0 = true_y[0]
true_y0 = torch.tensor([true_y0], dtype=torch.float32).to(device)
true_y = torch.tensor([[y] for y in true_y], dtype=torch.float32).to(device)


def get_batch():
    s = torch.from_numpy(
        np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs('png')
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, odefunc, itr):
    if args.viz:
        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1],
                     'g-')
        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', t.cpu().numpy(),
                     pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        ax_traj.set_ylim(-2, 2)
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_phase.set_xlim(-2, 2)
        ax_phase.set_ylim(-2, 2)

        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('x')
        ax_vecfield.set_ylabel('y')

        y, x = np.mgrid[-2:2:21j, -2:2:21j]
        dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)).cpu().detach().numpy()
        mag = np.sqrt(dydt[:, 0] ** 2 + dydt[:, 1] ** 2).reshape(-1, 1)
        dydt = (dydt / mag)
        dydt = dydt.reshape(21, 21, 2)

        ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        ax_vecfield.set_xlim(-2, 2)
        ax_vecfield.set_ylim(-2, 2)

        fig.tight_layout()
        plt.savefig('png/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(146, 13),
            nn.Tanh(),
            nn.Linear(13, 64),
        )

        self.net2 = nn.Sequential(
            nn.Linear(146, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
        )

        self.net3 = nn.Sequential(
            nn.Linear(146, 128),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        w, lv, rv, A, Ap, diff = torch.split(y, [1, 8, 8, 64, 64, 1], dim=-1)
        B = A.reshape(list(A.size())[:-1] + [8, 8])
        Bp = Ap.reshape(list(Ap.size())[:-1] + [8, 8])
        # print((B - Bp).size())
        # print(diff.size())
        w1 = torch.matmul(lv.unsqueeze(-2), torch.div((B - Bp), diff.unsqueeze(-1).expand_as(B - Bp)) * sampling_rate)
        w1 = torch.matmul(w1, rv.unsqueeze(-1))
        w1 = w1.squeeze(-1) / torch.matmul(lv, torch.transpose(rv, dim0=-1, dim1=-2))
        # if len(lv.size()) == 3:
        #     c = torch.einsum('ijk,ijk->ij', lv, rv)
        # else:
        #     c = torch.einsum('ij,ij->i', lv, rv)
        # c = c.unsqueeze(-1)
        # w1 = torch.div(w1, c)
        Ai = A.reshape(list(self.net(y).size())[:-1] + [8, 8])
        w2 = torch.matmul(rv.unsqueeze(-2), Ai)
        w2 = torch.matmul(w2, torch.div((B - Bp), diff.unsqueeze(-1).expand_as(B - Bp)) * sampling_rate)
        w2 = torch.matmul(w2, rv.unsqueeze(-1))
        w2 = w2.squeeze(-1)
        w2 = torch.matmul(w2, rv) - torch.matmul(torch.matmul(Ai, torch.div((B - Bp), diff.unsqueeze(-1).expand_as(B - Bp)) * sampling_rate), rv.unsqueeze(-1)).squeeze(-1)
        At = torch.transpose(Ai, dim0=-1, dim1=-2)
        # At = A.reshape(list(self.net2(y).size())[:-1] + [8, 8])
        w3 = torch.matmul(lv.unsqueeze(-2), At)
        w3 = torch.matmul(w3, torch.transpose(torch.div((B - Bp), diff.unsqueeze(-1).expand_as(B - Bp)) * sampling_rate, dim0=-1, dim1=-2))
        w3 = torch.matmul(w3, lv.unsqueeze(-1))
        w3 = w3.squeeze(-1)
        w3 = torch.matmul(w3, lv) - torch.matmul(torch.matmul(At, torch.transpose(torch.div((B - Bp), diff.unsqueeze(-1).expand_as(B - Bp)) * sampling_rate, dim0=-1, dim1=-2)), lv.unsqueeze(-1)).squeeze(-1)
        return torch.cat((w1, w3, w2, self.net3(y), diff), dim=-1)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


if __name__ == '__main__':

    ii = 0

    func = ODEFunc().to(device)

    optimizer = optim.RMSprop(func.parameters(), lr=1e-4)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)

    loss_meter = RunningAverageMeter(0.97)

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()
        pred_y = odeint(func, batch_y0, batch_t).to(device)
        loss = torch.mean(torch.square(pred_y[0: 5, :, 0] - batch_y[0: 5, :, 0]))
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = odeint(func, true_y0, t)
                loss = torch.mean(torch.square(pred_y[0: 5, :, 0] - true_y[0: 5, :, 0]))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                # print(pred_y[0: 20, 0, 0])
                # print(true_y[0: 20, 0, 0])
                visualize(true_y, pred_y, func, ii)
                ii += 1

        end = time.time()