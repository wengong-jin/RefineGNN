import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from structgen.utils import *


class MPNEncoder(nn.Module):
    
    def __init__(self, args, node_in, edge_in, direction='forward'):
        super(MPNEncoder, self).__init__()
        self.node_in, self.edge_in = node_in, edge_in
        self.direction = direction
        self.W_v = nn.Sequential(
                nn.Linear(self.node_in, args.hidden_size, bias=True),
                Normalize(args.hidden_size)
        )
        self.W_e = nn.Sequential(
                nn.Linear(self.edge_in, args.hidden_size, bias=True),
                Normalize(args.hidden_size)
        )
        self.W_s = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = nn.ModuleList([
                MPNNLayer(args.hidden_size, args.hidden_size * 3, dropout=args.dropout)
                for _ in range(args.depth)
        ])
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def autoregressive_mask(self, E_idx):
        N_nodes = E_idx.size(1)
        ii = torch.arange(N_nodes).cuda()
        ii = ii.view((1, -1, 1))
        mask = E_idx - ii < 0
        return mask.float()

    def forward(self, V, E, S, E_idx, mask):
        h_v = self.W_v(V)  # [B, N, H] 
        h_e = self.W_e(E)  # [B, N, K, H] 
        h_s = self.W_s(S)  # [B, N, H] 
        nei_s = gather_nodes(h_s, E_idx)  # [B, N, K, H]

        if self.direction == 'forward':
            vmask = self.autoregressive_mask(E_idx)  # [B, N, K]
            vmask = mask.unsqueeze(-1) * vmask
        elif self.direction == 'bidirectional':
            # [B, N, 1] -> [B, N, K, 1] -> [B, N, K]
            vmask = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        else:
            raise ValueError('invalid direction', self.direction)

        h = h_v
        for layer in self.layers:
            nei_v = gather_nodes(h, E_idx)  # [B, N, K, H]
            nei_h = torch.cat([nei_v, nei_s, h_e], dim=-1)
            h = layer(h, nei_h, mask_attend=vmask)  # [B, N, H]
            h = h * mask.unsqueeze(-1)  # [B, N, H]
        return h

    # incremental forward for unidirectional model
    def inc_forward(self, V, E, S, E_idx, mask, h_all, t):
        assert self.direction == 'forward'
        h_v = self.W_v(V[:, t:t+1])  # [B, 1, H] 
        h_e = self.W_e(E[:, t:t+1])  # [B, 1, K, H] 
        nei_s = gather_nodes(S.unsqueeze(-1), E_idx[:, t:t+1])  # [B, 1, K, 1]
        nei_s = self.W_s(nei_s.squeeze(-1))  # [B, 1, K, H]

        # sequence prediction
        h_all[0] = insert_tensor(h_all[0], h_v, t)  # h_all[0][:, t:t+1] = h_v
        for i, layer in enumerate(self.layers):
            nei_v = gather_nodes(h_all[i], E_idx[:, t:t+1])  # [B, 1, K, H]
            vmask = (E_idx[:, t:t+1] < t).float()  # [B, 1, K]
            nei_h = torch.cat([nei_v, nei_s, h_e], dim=-1)
            cur_h = h_all[i][:, t:t+1]
            h = layer(cur_h, nei_h, mask_attend=vmask)  # [B, 1, H]
            new_h = h * mask[:, t:t+1].unsqueeze(-1)  # [B, 1, H]
            h_all[i + 1] = insert_tensor(h_all[i + 1], new_h, t)
        return h_all

