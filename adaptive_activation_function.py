import torch
import torch.nn as nn
from torch import optim, autograd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import torch.nn.functional as F
from math import pi
import numpy as np
import sobol_seq as sobol

"""
-\Delta u = 0 in \Omega
u = r^{1/2}sin(theta/2) on Gamma
\Omega = (-1, 1)^2 \ (0,1)^2
exact solution: u = r^{1/2}sin(theta/2)
"""
class PowerReLU(nn.Module):
    """
    Implements simga(x)^(power)
    Applies a power of the rectified linear unit element-wise.
    NOTE: inplace may not be working.
    Can set inplace for inplace operation if desired.
    BUT I don't think it is working now.
    INPUT:
        x -- size (N,*) tensor where * is any number of additional
             dimensions
    OUTPUT:
        y -- size (N,*)
    """

    def __init__(self, power, inplace=False):
        super(PowerReLU, self).__init__()
        self.inplace = inplace
        self.power = power

    def forward(self, input):
        # y = F.relu(input, inplace=self.inplace)
        # return torch.pow(y, self.power)
        return torch.tanh(input * self.power * 10)


class Block(nn.Module):
    """
    IMplementation of the block used in the Deep Ritz
    Paper
    Parameters:
    in_N  -- dimension of the input
    width -- number of nodes in the interior middle layer
    out_N -- dimension of the output
    phi   -- activation function used
    """

    def __init__(self, in_N, width, out_N, a):
        super(Block, self).__init__()
        # create the necessary linear layers
        self.a = a
        self.L1 = nn.Linear(in_N, width)
        self.L2 = nn.Linear(width, out_N)
        # choose appropriate activation function
        self.phi = PowerReLU(power=self.a)

    def forward(self, x):
        return self.phi(self.L2(self.phi(self.L1(x)))) + x


class drrnn(nn.Module):
    """
    drrnn -- Deep Ritz Residual Neural Network
    Implements a network with the architecture used in the
    deep ritz method paper
    Parameters:
        in_N  -- input dimension
        out_N -- output dimension
        m     -- width of layers that form blocks
        depth -- number of blocks to be stacked
        phi   -- the activation function
    """

    def __init__(self, in_N, m, out_N, a, depth=4):
        super(drrnn, self).__init__()
        # set parameters
        self.in_N = in_N
        self.m = m
        self.a = a
        self.out_N = out_N
        self.depth = depth
        self.phi = PowerReLU(self.a)
        # list for holding all the blocks
        self.stack = nn.ModuleList()

        # add first layer to list
        self.stack.append(nn.Linear(in_N, m))

        # add middle blocks to list
        for i in range(depth):
            self.stack.append(Block(m, m, m, a))

        # add output linear layer
        self.stack.append(nn.Linear(m, out_N))

    def forward(self, x):
        # first layer
        for i in range(len(self.stack)):
            x = self.stack[i](x)
        return x


def get_interior_points(N=300):
    """
    randomly sample N points from interior of [-1,1]^d
    """
    return torch.rand(N, 1) * 6 - 3


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)


def function_u_exact(x):
    return (x ** 3 - x) * torch.sin(7 * x) / 7 + torch.sin(25 * x)

def main():

    epochs = 5000
    in_N = 1
    m = 20
    out_N = 1
    # a = torch.tensor([1.0])
    a = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
    print(torch.cuda.is_available())
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    model = drrnn(in_N, m, out_N, a).to(device)

    model.apply(weights_init)
    optimizer = optim.Adam([
        {'params': model.parameters()}], lr=1e-4)
    StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    print(model)

    best_loss, best_epoch = 1000, 0
    tt = time.time()
    xr = get_interior_points(300)
    exact_value = function_u_exact(xr)
    xr = xr.to(device)
    for epoch in range(epochs+1):

        # generate the data set
        output_r = model(xr)
        loss = torch.mean(torch.square(output_r - exact_value))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        StepLR.step()
        if epoch % 100 == 0:

            print(time.time() - tt, 'epoch:', epoch, 'loss:', loss.item(), a.item())
            tt = time.time()

    with torch.no_grad():
        x = torch.linspace(-3, 3, 1001).reshape(-1, 1)
        pred = model(x)
        exact = function_u_exact(x)
        err = torch.norm(pred - exact) / torch.norm(exact)
        print('err:', err)

    plt.figure()
    x = torch.linspace(-3, 3, 1001).reshape(-1, 1)
    exact_value = function_u_exact(x)
    plt.plot(x, exact_value, label='exact')
    pred_value = model(x).detach()
    plt.plot(x, pred_value, label='pred')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
