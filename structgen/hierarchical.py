import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from structgen.encoder import MPNEncoder
from structgen.data import alphabet
from structgen.utils import *
from structgen.protein_features import ProteinFeatures


class HierarchicalEncoder(nn.Module):
    
    def __init__(self, args, node_in, edge_in):
        super(HierarchicalEncoder, self).__init__()
        self.node_in, self.edge_in = node_in, edge_in
        self.W_v = nn.Sequential(
                nn.Linear(self.node_in, args.hidden_size, bias=True),
                Normalize(args.hidden_size)
        )
        self.W_e = nn.Sequential(
                nn.Linear(self.edge_in, args.hidden_size, bias=True),
                Normalize(args.hidden_size)
        )
        self.layers = nn.ModuleList([
                MPNNLayer(args.hidden_size, args.hidden_size * 3, dropout=args.dropout)
                for _ in range(args.depth)
        ])
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, V, E, hS, E_idx, mask):
        h_v = self.W_v(V)  # [B, N, H] 
        h_e = self.W_e(E)  # [B, N, K, H] 
        nei_s = gather_nodes(hS, E_idx)  # [B, N, K, H]

        # [B, N, 1] -> [B, N, K, 1] -> [B, N, K]
        vmask = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        h = h_v
        for layer in self.layers:
            nei_v = gather_nodes(h, E_idx)  # [B, N, K, H]
            nei_h = torch.cat([nei_v, nei_s, h_e], dim=-1)
            h = layer(h, nei_h, mask_attend=vmask)  # [B, N, H]
            h = h * mask.unsqueeze(-1)  # [B, N, H]
        return h


class HierarchicalDecoder(nn.Module):

    def __init__(self, args):
        super(HierarchicalDecoder, self).__init__()
        self.cdr_type = args.cdr_type
        self.k_neighbors = args.k_neighbors
        self.block_size = args.block_size
        self.update_freq = args.update_freq
        self.hidden_size = args.hidden_size
        self.pos_embedding = PosEmbedding(16)
        
        self.features = ProteinFeatures(
                top_k=args.k_neighbors, num_rbf=args.num_rbf,
                features_type='full',
                direction='bidirectional'
        )
        self.node_in, self.edge_in = self.features.feature_dimensions['full']
        self.O_d0 = nn.Linear(args.hidden_size, 12)
        self.O_d = nn.Linear(args.hidden_size, 12)
        self.O_s = nn.Linear(args.hidden_size, args.vocab_size)
        self.W_s = nn.Embedding(args.vocab_size, args.hidden_size)

        self.struct_mpn = HierarchicalEncoder(args, self.node_in, self.edge_in)
        self.seq_mpn = HierarchicalEncoder(args, self.node_in, self.edge_in)
        self.init_mpn = HierarchicalEncoder(args, 16, 32)
        self.rnn = nn.GRU(
                args.hidden_size, args.hidden_size, batch_first=True, 
                num_layers=1, bidirectional=True
        ) 
        self.W_stc = nn.Sequential(
                nn.Linear(args.hidden_size * 2, args.hidden_size),
                nn.ReLU(),
        )
        self.W_seq = nn.Sequential(
                nn.Linear(args.hidden_size * 2, args.hidden_size),
                nn.ReLU(),
        )

        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.huber_loss = nn.SmoothL1Loss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def init_struct(self, B, N, K):
        # initial V
        pos = torch.arange(N).cuda()
        V = self.pos_embedding(pos.view(1, N, 1))  # [1, N, 1, 16]
        V = V.squeeze(2).expand(B, -1, -1)  # [B, N, 6]
        # initial E_idx
        pos = pos.unsqueeze(0) - pos.unsqueeze(1)     # [N, N]
        D_idx, E_idx = pos.abs().topk(k=K, dim=-1, largest=False)    # [N, K]
        E_idx = E_idx.unsqueeze(0).expand(B, -1, -1)  # [B, N, K]
        D_idx = D_idx.unsqueeze(0).expand(B, -1, -1)  # [B, N, K]
        # initial E
        E_rbf = self.features._rbf(3 * D_idx)
        E_pos = self.features.embeddings(E_idx)
        E = torch.cat((E_pos, E_rbf), dim=-1)
        return V, E, E_idx

    def init_coords(self, S, mask):
        B, N = S.size(0), S.size(1)
        K = min(self.k_neighbors, N)
        V, E, E_idx = self.init_struct(B, N, K)
        h = self.init_mpn(V, E, S, E_idx, mask)
        return self.predict_dist(self.O_d0(h))

    # Q: [B, N, H], K, V: [B, M, H]
    def attention(self, Q, context, cmask, W):
        att = torch.bmm(Q, context.transpose(1, 2))  # [B, N, M]
        att = att - 1e6 * (1 - cmask.unsqueeze(1))
        att = F.softmax(att, dim=-1)
        out = torch.bmm(att, context)  # [B, N, M] * [B, M, H]
        out = torch.cat([Q, out], dim=-1)
        return W(out)

    def predict_dist(self, X):
        X = X.view(X.size(0), X.size(1), 4, 3)
        X_ca = X[:, :, 1, :]
        dX = X_ca[:, None, :, :] - X_ca[:, :, None, :]
        D = torch.sum(dX ** 2, dim=-1)
        V = self.features._dihedrals(X)
        AD = self.features._AD_features(X[:,:,1,:])
        return X.detach().clone(), D, V, AD

    def mask_mean(self, X, mask, i):
        # [B, N, 4, 3] -> [B, 1, 4, 3] / [B, 1, 1, 1]
        X = X[:, i:i+self.block_size]
        if X.dim() == 4:
            mask = mask[:, i:i+self.block_size].unsqueeze(-1).unsqueeze(-1)
        else:
            mask = mask[:, i:i+self.block_size].unsqueeze(-1)
        return torch.sum(X * mask, dim=1, keepdims=True) / (mask.sum(dim=1, keepdims=True) + 1e-8)

    def make_X_blocks(self, X, l, r, mask):
        N = X.size(1)
        lblocks = [self.mask_mean(X, mask, i) for i in range(0, l, self.block_size)]
        rblocks = [self.mask_mean(X, mask, i) for i in range(r + 1, N, self.block_size)]
        bX = torch.cat(lblocks + [X[:, l:r+1]] + rblocks, dim=1)
        return bX.detach()

    def make_S_blocks(self, LS, S, RS, l, r, mask):
        N = S.size(1)
        hS = self.W_s(S)
        LS = [self.mask_mean(hS, mask, i) for i in range(0, l, self.block_size)]
        RS = [self.mask_mean(hS, mask, i) for i in range(r + 1, N, self.block_size)]
        bS = torch.cat(LS + [hS[:, l:r+1]] + RS, dim=1)
        lmask = [mask[:, i:i+self.block_size].amax(dim=1, keepdims=True) for i in range(0, l, self.block_size)]
        rmask = [mask[:, i:i+self.block_size].amax(dim=1, keepdims=True) for i in range(r + 1, N, self.block_size)]
        bmask = torch.cat(lmask + [mask[:, l:r+1]] + rmask, dim=1)
        return bS, bmask, len(LS), len(RS)

    def get_completion_mask(self, B, N, cdr_range):
        cmask = torch.zeros(B, N).cuda()
        for i, (l,r) in enumerate(cdr_range):
            cmask[i, l:r+1] = 1
        return cmask

    def remove_cdr_coords(self, X, cdr_range):
        X = X.clone()
        for i, (l,r) in enumerate(cdr_range):
            X[i, l:r+1, :, :] = 0
        return X.clone()

    def forward(self, true_X, true_S, true_cdr, mask):
        B, N = mask.size(0), mask.size(1)
        K = min(self.k_neighbors, N)

        cdr_range = [(cdr.index(self.cdr_type), cdr.rindex(self.cdr_type)) for cdr in true_cdr]
        T_min = min([l for l,r in cdr_range])
        T_max = max([r for l,r in cdr_range])
        cmask = self.get_completion_mask(B, N, cdr_range)
        smask = mask.clone()

        # make blocks and encode framework
        S = true_S.clone() * (1 - cmask.long())
        hS, _ = self.rnn(self.W_s(S))
        LS, RS = hS[:, :, :self.hidden_size], hS[:, :, self.hidden_size:]
        hS, mask, offset, suffix = self.make_S_blocks(LS, S, RS, T_min, T_max, mask)
        cmask = torch.cat([cmask.new_zeros(B, offset), cmask[:, T_min:T_max+1], cmask.new_zeros(B, suffix)], dim=1)

        # Ground truth 
        true_X = self.make_X_blocks(true_X, T_min, T_max, smask)
        true_V = self.features._dihedrals(true_X)
        true_AD = self.features._AD_features(true_X[:,:,1,:])
        true_D, mask_2D = pairwise_distance(true_X, mask)
        true_D = true_D ** 2

        # initial loss
        sloss = 0.
        X, D, V, AD = self.init_coords(hS, mask)
        X = X.detach().clone()
        dloss = self.huber_loss(D, true_D)
        vloss = self.mse_loss(V, true_V)
        aloss = self.mse_loss(AD, true_AD)

        for t in range(T_min, T_max + 1):
            # Prepare input
            V, E, E_idx = self.features(X, mask)
            hS = self.make_S_blocks(LS, S, RS, T_min, T_max, smask)[0]

            # Predict residue t
            h = self.seq_mpn(V, E, hS, E_idx, mask)
            h = self.attention(h, LS, smask, self.W_seq)
            logits = self.O_s(h[:, offset + t - T_min])
            snll = self.ce_loss(logits, true_S[:, t])
            sloss = sloss + torch.sum(snll * cmask[:, offset + t - T_min])

            # Teacher forcing on S
            S = S.clone()
            S[:, t] = true_S[:, t]
            S = S.clone()

            # Iterative refinement
            if t % self.update_freq == 0:
                h = self.struct_mpn(V, E, hS, E_idx, mask)
                h = self.attention(h, LS, smask, self.W_stc)
                X, D, V, AD = self.predict_dist(self.O_d(h))
                X = X.detach().clone()
                dloss = dloss + self.huber_loss(D, true_D)
                vloss = vloss + self.mse_loss(V, true_V)
                aloss = aloss + self.mse_loss(AD, true_AD)

        dloss = torch.sum(dloss * mask_2D) / mask_2D.sum()
        vloss = torch.sum(vloss * mask.unsqueeze(-1)) / mask.sum()
        aloss = torch.sum(aloss * mask.unsqueeze(-1)) / mask.sum()
        sloss = sloss.sum() / cmask.sum()
        loss = sloss + dloss + vloss + aloss
        return loss, sloss

    def log_prob(self, true_S, true_cdr, mask):
        B, N = mask.size(0), mask.size(1)
        K = min(self.k_neighbors, N)

        cdr_range = [(cdr.index(self.cdr_type), cdr.rindex(self.cdr_type)) for cdr in true_cdr]
        T_min = min([l for l,r in cdr_range])
        T_max = max([r for l,r in cdr_range])
        cmask = self.get_completion_mask(B, N, cdr_range)
        smask = mask.clone()

        # initialize
        S = true_S.clone() * (1 - cmask.long())
        hS, _ = self.rnn(self.W_s(S))
        LS, RS = hS[:, :, :self.hidden_size], hS[:, :, self.hidden_size:]
        hS, mask, offset, suffix = self.make_S_blocks(LS, S, RS, T_min, T_max, mask)
        cmask = torch.cat([cmask.new_zeros(B, offset), cmask[:, T_min:T_max+1], cmask.new_zeros(B, suffix)], dim=1)

        sloss = 0.
        X = self.init_coords(hS, mask)[0]
        X = X.detach().clone()

        for t in range(T_min, T_max + 1):
            # Prepare input
            V, E, E_idx = self.features(X, mask)
            hS = self.make_S_blocks(LS, S, RS, T_min, T_max, smask)[0]

            # Predict residue t
            h = self.seq_mpn(V, E, hS, E_idx, mask)
            h = self.attention(h, LS, smask, self.W_seq)
            logits = self.O_s(h[:, offset + t - T_min])
            snll = self.ce_loss(logits, true_S[:, t])
            sloss = sloss + snll * cmask[:, offset + t - T_min]

            # Teacher forcing on S
            S = S.clone()
            S[:, t] = true_S[:, t]
            S = S.clone()

            # Iterative refinement
            if t % self.update_freq == 0:
                h = self.struct_mpn(V, E, hS, E_idx, mask)
                h = self.attention(h, LS, smask, self.W_stc)
                X = self.predict_dist(self.O_d(h))[0]
                X = X.detach().clone()

        ppl = sloss / cmask.sum(dim=-1)
        sloss = sloss.sum() / cmask.sum()
        return ReturnType(nll=sloss, ppl=ppl, X=X, X_cdr=X[:, offset:offset+T_max-T_min+1])

    def generate(self, true_S, true_cdr, mask, return_ppl=False):
        B, N = mask.size(0), mask.size(1)
        K = min(self.k_neighbors, N)

        cdr_range = [(cdr.index(self.cdr_type), cdr.rindex(self.cdr_type)) for cdr in true_cdr]
        T_min = min([l for l,r in cdr_range])
        T_max = max([r for l,r in cdr_range])
        cmask = self.get_completion_mask(B, N, cdr_range)
        smask = mask.clone()

        # initialize
        S = true_S.clone() * (1 - cmask.long())
        hS, _ = self.rnn(self.W_s(S))
        LS, RS = hS[:, :, :self.hidden_size], hS[:, :, self.hidden_size:]
        hS, mask, offset, suffix = self.make_S_blocks(LS, S, RS, T_min, T_max, mask)
        cmask = torch.cat([cmask.new_zeros(B, offset), cmask[:, T_min:T_max+1], cmask.new_zeros(B, suffix)], dim=1)

        X = self.init_coords(hS, mask)[0]
        X = X.detach().clone()
        sloss = 0

        for t in range(T_min, T_max + 1):
            # Prepare input
            V, E, E_idx = self.features(X, mask)
            hS = self.make_S_blocks(LS, S, RS, T_min, T_max, smask)[0]

            # Predict residue t
            h = self.seq_mpn(V, E, hS, E_idx, mask)
            h = self.attention(h, LS, smask, self.W_seq)
            logits = self.O_s(h[:, offset + t - T_min])
            prob = F.softmax(logits, dim=-1)  # [B, 20]
            S[:, t] = torch.multinomial(prob, num_samples=1).squeeze(-1)  # [B, 1]
            sloss = sloss + self.ce_loss(logits, S[:, t]) * cmask[:, offset + t - T_min]

            # Iterative refinement
            h = self.struct_mpn(V, E, hS, E_idx, mask)
            h = self.attention(h, LS, smask, self.W_stc)
            X = self.predict_dist(self.O_d(h))[0]
            X = X.detach().clone()

        S = S.tolist()
        S = [''.join([alphabet[S[i][j]] for j in range(cdr_range[i][0], cdr_range[i][1] + 1)]) for i in range(B)]
        ppl = torch.exp(sloss / cmask.sum(dim=-1))
        return (S, ppl, X[:, offset:offset+T_max-T_min+1]) if return_ppl else S
