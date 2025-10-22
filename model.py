import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, x_dim=3, o_dim=0, hidden=128):
        super().__init__()
        in_dim = x_dim + 1 + o_dim   
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, x_dim)   
        )

    def forward(self, x_t, s, o_t=None):
        
        if s.dim() == 1: s = s.unsqueeze(-1)   # [B,1]
        feats = [x_t, s]
        if o_t is not None:
            if o_t.dim() == 1: o_t = o_t.unsqueeze(-1)
            feats.append(o_t)
        h = torch.cat(feats, dim=-1)
        return self.net(h)           

        
        