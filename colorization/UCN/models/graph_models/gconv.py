import torch
import torch.nn as nn


class GC(nn.Module):
    def __init__(self, in_features, out_features):
        super(GC, self).__init__()
        self.num_inputs = in_features
        self.num_outputs = out_features

        self.a_fc = nn.Linear(self.num_inputs, self.num_outputs)
        self.u_fc = nn.Linear(self.num_inputs, self.num_outputs)

    def forward(self, adj, x, norm=True):
        if norm is True:
            adj = adj / torch.norm(adj, p=1, dim=1, keepdim=True)

        ax = self.a_fc(x)
        ux = self.u_fc(x)

        out = torch.bmm(adj, ax) + ux
        return out
