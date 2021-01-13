import torch
import torch.nn as nn
from torch import optim, autograd
from matplotlib import pyplot as plt
from math import pi
import numpy as np


class Unit(nn.Module):

    def __init__(self, in_N, out_N):
        super(Unit, self).__init__()
        self.in_N = in_N
        self.out_N = out_N
        self.L = nn.Linear(in_N, out_N, bias=True)

    def forward(self, x):
        x1 = self.L(x)
        x2 = torch.tanh(x1)
        return x2


class NN(nn.Module):

    def __init__(self, in_N, width, depth, out_N):
        super(NN, self).__init__()
        self.width = width
        self.in_N = in_N
        self.out_N = out_N
        self.stack = nn.ModuleList()

        self.stack.append(Unit(in_N, width))

        for i in range(depth-1):
            self.stack.append(Unit(width, width))

        self.stack.append(nn.Linear(width, out_N, bias=False))

    def forward(self, x):
        # first layer
        for i in range(len(self.stack)):
            x = self.stack[i](x)
        return x

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)


def main():
    x = torch.rand(50, 1) * 4 * pi - 2 * pi
    noise = torch.normal(0, 0.1, [50, 1])
    y = torch.cos(x) + noise
    model = NN(1, 20, 2, 1)
    model.apply(weights_init)

    optimizer = optim.Adam([{'params': model.parameters()}], lr=1e-2)
    StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    for epoch in range(1000):
        pred = (model(x) + model(-x)) / 2
        loss = torch.mean(torch.square(y - pred))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        StepLR.step()

        if epoch % 20 == 0:
            print('epoch:', epoch, 'loss:', loss.item())

    with torch.no_grad():
        xx = torch.linspace(-2*pi, 2*pi, 200).reshape(-1, 1)
        pred = (model(xx) + model(-xx)) / 2
        plt.figure()
        plt.plot(xx, pred)
        plt.scatter(x, y)
        plt.show()


if __name__ == '__main__':
    main()
