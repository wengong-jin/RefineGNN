import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple

ReturnType = namedtuple('ReturnType',('nll','ppl','X','X_cdr'), defaults=(None, None, None, None))

# problem: hard to add mask in SVD
def kabsch(A, B):
    a_mean = A.mean(dim=1, keepdims=True)
    b_mean = B.mean(dim=1, keepdims=True)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = torch.bmm(A_c.transpose(1,2), B_c)  # [B, 3, 3]
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = torch.bmm(V, U.transpose(1,2))  # [B, 3, 3]
    # Translation vector
    t = b_mean - torch.bmm(R, a_mean.transpose(1,2)).transpose(1,2)
    A_aligned = torch.bmm(R, A.transpose(1,2)).transpose(1,2) + t
    return A_aligned, R, t

# A: [B, N, 3], B: [B, N, 3], mask: [B, N]
def compute_rmsd(A, B, mask):
    A_aligned, _, _ = kabsch(A, B)
    rmsd = ((A_aligned - B) ** 2).sum(dim=-1)
    rmsd = torch.sum(rmsd * mask, dim=-1) / (mask.sum(dim=-1) + 1e-6)
    return rmsd.sqrt()

def autoregressive_mask(E_idx):
    N_nodes = E_idx.size(1)
    ii = torch.arange(N_nodes).cuda()
    ii = ii.view((1, -1, 1))
    mask = E_idx - ii < 0
    return mask.float()

def fit_coords(D, E_idx, mask, lr=2.0, num_steps=200):
    with torch.enable_grad():
        pred_xyz = torch.randn(D.size(0), D.size(1), 3).cuda()
        pred_xyz = pred_xyz.requires_grad_()
        optimizer = torch.optim.Adam([pred_xyz], lr=lr)
        vmask = autoregressive_mask(E_idx) * mask.unsqueeze(-1)

        for _ in range(num_steps):
            optimizer.zero_grad()
            cur_D = (pred_xyz.unsqueeze(2) - pred_xyz.unsqueeze(1)).norm(dim=-1, p=2)  # [B, N, N]
            cur_D = gather_edges(cur_D.unsqueeze(-1), E_idx).squeeze(-1)
            loss = (D - cur_D) ** 2
            loss = torch.sum(loss * vmask) / vmask.sum()
            loss.backward()
            optimizer.step()

    return pred_xyz.detach().clone()

# The following gather functions
def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features

def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn

def gather_2d(x, idx):
    # x: [B, N], idx: [B, K]
    batch_size, nei_size = idx.size(0), idx.size(1)
    idx_flat1 = idx.reshape(-1)
    idx_flat0 = torch.cat([torch.arange(batch_size)] * nei_size, dim=0)
    new_x = x[idx_flat0, idx_flat1]
    return new_x.view(batch_size, nei_size)

# h: [B, N, H], x: [B, 1, H]
def insert_tensor(h, x, t):
    if t == 0:
        return torch.cat((x, h[:, 1:]), dim=1)
    elif t == h.size(1) - 1:
        return torch.cat((h[:, :-1], x), dim=1)
    else:
        return torch.cat((h[:, :t], x, h[:, t+1:]), dim=1)

def pairwise_distance(X, mask):
    X_ca = X[:, :, 1, :]  # alpha carbon
    mask_2D = mask.unsqueeze(1) * mask.unsqueeze(2)
    dX = X_ca.unsqueeze(1) - X_ca.unsqueeze(2)
    D = mask_2D * torch.sqrt(torch.sum(dX**2, dim=3))
    return D, mask_2D

def get_nei_label(X, mask, K):
    D, mask_2D = pairwise_distance(X, mask)
    nmask = torch.arange(X.size(1)).cuda()
    nmask = nmask.view(1,-1,1) > nmask.view(1,1,-1)
    nmask = nmask.float() * mask_2D  # [B, N, N]

    # Identify k nearest neighbors (including self)
    D_adjust = D + (1. - nmask) * 100000
    D_sorted, E_idx = torch.topk(D_adjust, D.size(1), dim=-1, largest=False)
    E_next = E_idx[:, 1:]
    D_next = D_sorted[:, 1:]
    nmask = gather_edges(nmask.unsqueeze(-1), E_next)  # [B, N-1, N, 1]
    nlabel = torch.zeros_like(E_next)
    nlabel[:, :, :K] = 1
    return E_next, nlabel, D_next, nmask.squeeze(-1)


class Normalize(nn.Module):

    def __init__(self, features, epsilon=1e-6):
        super(Normalize, self).__init__()
        self.gain = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x, dim=-1):
        mu = x.mean(dim, keepdim=True)
        sigma = torch.sqrt(x.var(dim, keepdim=True) + self.epsilon)
        gain = self.gain
        bias = self.bias
        # Reshape
        if dim != -1:
            shape = [1] * len(mu.size())
            shape[dim] = self.gain.size()[0]
            gain = gain.view(shape)
            bias = bias.view(shape)

        return gain * (x - mu) / (sigma + self.epsilon) + bias


class MPNNLayer(nn.Module):

    def __init__(self, num_hidden, num_in, dropout):
        super(MPNNLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.dropout = nn.Dropout(dropout)
        self.norm = Normalize(num_hidden)
        self.W = nn.Sequential(
                nn.Linear(num_hidden + num_in, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, num_hidden),
        )

    def forward(self, h_V, h_E, mask_attend):
        # h_V: [B, N, H]; h_E: [B, N, K, H]
        # mask_attend: [B, N, K]
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E], dim=-1)  # [B, N, K, H]
        h_message = self.W(h_EV) * mask_attend.unsqueeze(-1)
        dh = torch.mean(h_message, dim=-2)
        h_V = self.norm(h_V + self.dropout(dh))
        return h_V


class FrameMPNNLayer(nn.Module):

    def __init__(self, num_hidden, num_in, dropout):
        super(FrameMPNNLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.dropout = nn.Dropout(dropout)
        self.norm = Normalize(num_hidden)
        self.W = nn.Sequential(
                nn.Linear(num_hidden + num_in, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, num_hidden),
        )
        self.U = nn.Sequential(
                nn.Linear(num_hidden + num_in, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, num_hidden),
        )

    def forward(self, h_V, h_E2, h_E3, mask2, mask3):
        # h_V: [B, N, H]; h_E: [B, N, K, H]
        # mask_attend: [B, N, K]
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E2.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E2], dim=-1)  # [B, N, K, H]
        h_message = self.W(h_EV) * mask2.unsqueeze(-1)
        dh = torch.mean(h_message, dim=-2)

        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E3.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E3], dim=-1)  # [B, N, K, H]
        mask3 = mask3[:,:,:-1] * mask3[:,:,1:]
        u_message = self.U(h_EV) * mask3.unsqueeze(-1)
        du = torch.mean(u_message, dim=-2)

        h_V = self.norm(h_V + self.dropout(dh) + self.dropout(du))
        return h_V


class PosEmbedding(nn.Module):

    def __init__(self, num_embeddings):
        super(PosEmbedding, self).__init__()
        self.num_embeddings = num_embeddings

    def forward(self, E_idx):
        # Original Transformer frequencies
        frequency = torch.exp(
            torch.arange(0, self.num_embeddings, 2, dtype=torch.float32)
            * -(np.log(10000.0) / self.num_embeddings)
        ).cuda()
        angles = E_idx.unsqueeze(-1) * frequency.view((1,1,1,-1))
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E


if __name__ == "__main__":
    # Test RMSD calculation
    A = torch.tensor([[1., 1.], [2., 2.], [1.5, 3.]], dtype=torch.float32)
    R0 = torch.tensor([[np.cos(60), -np.sin(60)], [np.sin(60), np.cos(60)]], dtype=torch.float32)
    B = (R0.mm(A.T)).T
    t0 = torch.tensor([3., 3.])
    B += t0
    C = B * torch.tensor([-1., 1.])
    rmsd = compute_rmsd(
            torch.stack([A, A], dim=0),
            torch.stack([B, C], dim=0),
            mask=torch.ones(2, 3)
    )
    print(rmsd)

