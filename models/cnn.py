import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import pdb
import numpy as np
from math import sqrt

from models import fcn

class RoastConv2d(nn.Module):
    def __init__(self, in_channels,
                    out_channels,
                    kernel_size,
                    is_global,
                    weight=None,
                    init_scale=None,
                    compression=None,
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=1, 
                    bias=True):

        super(RoastConv2d, self).__init__()
        
        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size)
        
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.compression = compression
        self.groups = groups
        self.is_bias = bias
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.W_shape = (out_channels, int(in_channels/groups), kernel_size[0], kernel_size[1])
        
        self.wsize = int(np.prod(self.W_shape) * compression)
        self.weight = nn.Parameter(torch.zeros(self.wsize, dtype=torch.float), requires_grad=True)

        k = 1.0 * groups / (in_channels * np.prod(kernel_size))
        nn.init.uniform_(self.weight.data, a=-sqrt(k) , b = sqrt(k) )
        
        self.IDX = nn.Parameter(torch.randint(0, self.wsize, size=self.W_shape, dtype=torch.int64), requires_grad=False)
        self.G = nn.Parameter(torch.randint(0, 2, size=self.W_shape, dtype=torch.float)*2 - 1, requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
            
    def forward(self, x):
        W = torch.mul(self.weight[self.IDX], self.G)
        x = torch.reshape(x, [-1, 1, 28, 28])
        x = torch.nn.functional.conv2d(x, W, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return x

    def grad_comp_to_orig(self, grad):
        return torch.mul(grad[self.IDX], self.G)

    def grad_orig_to_comp(self, grad):
        out_grad = torch.zeros(
            self.wsize, dtype=torch.float, device=grad.device)
        out_grad.scatter_add_(0, self.IDX.view(-1), (torch.mul(grad, self.G)).view(-1))

        count = torch.zeros(self.wsize, dtype=torch.float, device=grad.device)
        count.scatter_add_(0, self.IDX.view(-1), torch.ones_like(self.IDX, device=grad.device, dtype=torch.float).view(-1))
        return torch.div(out_grad, count + 1e-3)

    def wt_comp_to_orig(self, wt):
        return torch.mul(wt[self.IDX], self.G)

    def wt_orig_to_comp(self, wt):
        out_wt = torch.zeros(self.wsize, dtype=torch.float, device=wt.device)
        out_wt.scatter_add_(0, self.IDX.view(-1), (torch.mul(wt, self.G)).view(-1))

        count = torch.zeros(self.wsize, dtype=torch.float, device=wt.device)
        count.scatter_add_(0, self.IDX.view(-1), torch.ones_like(self.IDX, device=wt.device, dtype=torch.float).view(-1)) + 1e-3
        return torch.div(out_wt, count + 1e-3)

    def __repr__(self):
        return "RoastConv2D(in_channels={}, out_channels={}, compression={}, kernel_size={}, stride={})".format(self.in_channels, self.out_channels, self.compression, self.kernel_size, self.stride)


# Conv2D ROAST Model
class ROASTCNN(fcn.BaseModel):
    def __init__(self, in_channels,
                    out_channels,
                    num_layers,
                    hidden_size,
                    num_class,
                    kernel_size,
                    compression,
                    seed):

        super(ROASTCNN, self).__init__()
        torch.manual_seed(seed)
        self.seed = seed

        # Define 2D convolutional layers
        self.first_layer = RoastConv2d(in_channels=1, out_channels=out_channels, kernel_size=kernel_size, is_global=False, compression=compression)
        self.first_layer_relu = nn.ReLU()
        # self.first_layer_maxpool = nn.MaxPool2d(kernel_size=2)
        mid_layers = []
        for i in range(num_layers - 3):
            mid_layers.append(RoastConv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, is_global=False, compression=compression))
            mid_layers.append(nn.ReLU())
            # mid_layers.append(nn.MaxPool2d(kernel_size=2))
        self.mid_layers = nn.Sequential(*mid_layers)
        self.flatten = nn.Flatten()

        # Define first fully connected layer
        self.fc1 = fcn.RoastLinear(11520, hidden_size, compression)
        self.fc1_relu = nn.ReLU()

        # Define second fully connected layer that outputs labels
        self.last_layer = fcn.RoastLinear(hidden_size, num_class, compression)
        
    def forward(self, x):
        x = self.first_layer(x)
        x = self.first_layer_relu(x)
        #x = self.first_layer_maxpool
        for layer in self.mid_layers:
            x = layer(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc1_relu(x)
        x = self.last_layer(x)
        x = F.log_softmax(x, dim=1)
        return x