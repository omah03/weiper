
import torch
import torch.nn as nn
import torch.nn.functional as F

class NormalizedRandomConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, bias=False):
        super(NormalizedRandomConv, self).__init__()
        if padding is None:
            padding = kernel_size // 2  

        weights = torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        with torch.no_grad():
            norms = weights.view(out_channels, -1).norm(dim=1, keepdim=True)
            weights = weights / norms.view(out_channels, 1, 1, 1)

        self.weights = nn.Parameter(weights, requires_grad=False)
        self.stride = stride
        self.padding = padding
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels,))

    def forward(self, x):
        return F.conv2d(x, self.weights, bias=self.bias, stride=self.stride, padding=self.padding)


def create_random_conv_projections(in_channels, out_channels, num_projections=5, kernel_size=3):
    # Returns a nn.ModuleList of NormalizedRandomConv modules
    projections = []
    for _ in range(num_projections):
        proj = NormalizedRandomConv(in_channels, out_channels, kernel_size=kernel_size)
        projections.append(proj)
    return nn.ModuleList(projections)
