import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from structgen.encoder import MPNEncoder
from structgen.data import alphabet
from structgen.utils import *
from structgen.protein_features import ProteinFeatures


class RevisionDecoder(nn.Module):

    def __init__(self, args):
        super(RevisionDecoder, self).__init__()
        self.k_neighbors = args.k_neighbors
        self.hidden_size = args.hidden_size
        self.pos_embedding = PosEmbedding(16)
        self.context = args.context

        self.features = ProteinFeatures(
                top_k=args.k_neighbors, num_rbf=args.num_rbf,
                features_type='full',
                direction='bidirectional'
        )
        self.node_in, self.edge_in = self.features.feature_dimensions['full']
        self.O_d0 = nn.Linear(args.hidden_size, 12)
        self.O_d = nn.Linear(args.hidden_size, 12)
        self.O_s = nn.Linear(args.hidden_size, args.vocab_size)

        self.struct_mpn = MPNEncoder(args, self.node_in, self.edge_in, direction='bidirectional')
        self.seq_mpn = MPNEncoder(args, self.node_in, self.edge_in, direction='bidirectional')

        if args.context:
            self.crnn = nn.GRU(
                    len(alphabet), args.hidden_size, 
                    batch_first=True, num_layers=1,
                    dropout=args.dropout
            )
            self.W_stc = nn.Sequential(
                    nn.Linear(args.hidden_size * 2, args.hidden_size),
                    nn.ReLU(),
            )
            self.W_seq = nn.Sequential(
                    nn.Linear(args.hidden_size * 2, args.hidden_size),
                    nn.ReLU(),
            )
        else:
            self.init_mpn = MPNEncoder(args, 16, 32, direction='bidirectional')
            self.W_stc = self.W_seq = None

        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.huber_loss = nn.SmoothL1Loss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def init_struct(self, B, N, K):
        # initial S and V
        S = torch.zeros(B, N).cuda().long()
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
        return V, E, S, E_idx

    def encode_context(self, cS, cmask, crange):
        cS = F.one_hot(cS, num_classes=len(alphabet)).float()
        cH, _ = self.crnn(cS)
        max_len = max([right - left + 1 for left,right in crange])
        cdr_h = [cH[i, left:right+1] for i,(left,right) in enumerate(crange)]
        cdr_h = [F.pad(h, (0,0,0,max_len-len(h))) for h in cdr_h]
        return torch.stack(cdr_h, dim=0), cH, cmask, crange

    def init_coords(self, mask):
        B, N = mask.size(0), mask.size(1)
        K = min(self.k_neighbors, N)
        V, E, S, E_idx = self.init_struct(B, N, K)
        h = self.init_mpn(V, E, S, E_idx, mask)
        return self.predict_dist(self.O_d0(h))

    # Q: [B, N, H], K, V: [B, M, H]
    def attention(self, Q, context, cmask, W):
        if self.context:
            att = torch.bmm(Q, context.transpose(1, 2))  # [B, N, M]
            att = att - 1e6 * (1 - cmask.unsqueeze(1))
            att = F.softmax(att, dim=-1)
            out = torch.bmm(att, context)  # [B, N, M] * [B, M, H]
            out = torch.cat([Q, out], dim=-1)
            return W(out)
        else:
            return Q

    def predict_dist(self, X):
        X = X.view(X.size(0), X.size(1), 4, 3)
        X_ca = X[:, :, 1, :]
        dX = X_ca[:, None, :, :] - X_ca[:, :, None, :]
        D = torch.sum(dX ** 2, dim=-1)
        V = self.features._dihedrals(X)
        AD = self.features._AD_features(X[:,:,1,:])
        return X.detach().clone(), D, V, AD

    def forward(self, true_X, true_S, L, mask, context=None):
        B, N = mask.size(0), mask.size(1)
        K = min(self.k_neighbors, N)

        # Ground truth 
        true_V = self.features._dihedrals(true_X)
        true_AD = self.features._AD_features(true_X[:,:,1,:])
        true_D, mask_2D = pairwise_distance(true_X, mask)
        true_D = true_D ** 2

        # initialize
        sloss = 0.
        S = torch.zeros(B, N).cuda().long()
        if self.context:
            h, cH, cmask, crange = self.encode_context(*context)
            X, D, V, AD = self.predict_dist(self.O_d0(h))
        else:
            X, D, V, AD = self.init_coords(mask)
            cH = cmask = None

        dloss = self.huber_loss(D, true_D)
        vloss = self.mse_loss(V, true_V)
        aloss = self.mse_loss(AD, true_AD)

        for t in range(N):
            # Predict residue t
            V, E, E_idx = self.features(X, mask)
            h = self.seq_mpn(V, E, S, E_idx, mask)
            h = self.attention(h, cH, cmask, self.W_seq)
            logits = self.O_s(h[:, t])
            snll = self.ce_loss(logits, true_S[:, t])
            sloss = sloss + torch.sum(snll * mask[:, t])

            # Teacher forcing on S
            S = S.clone()
            S[:, t] = true_S[:, t]
            S = S.clone()

            # Iterative refinement
            h = self.struct_mpn(V, E, S, E_idx, mask)
            h = self.attention(h, cH, cmask, self.W_stc)
            X, D, V, AD = self.predict_dist(self.O_d(h))
            dloss = dloss + self.huber_loss(D, true_D)
            vloss = vloss + self.mse_loss(V, true_V)
            aloss = aloss + self.mse_loss(AD, true_AD)

        dloss = torch.sum(dloss * mask_2D) / mask_2D.sum()
        vloss = torch.sum(vloss * mask.unsqueeze(-1)) / mask.sum()
        aloss = torch.sum(aloss * mask.unsqueeze(-1)) / mask.sum()
        sloss = sloss.sum() / mask.sum()
        return sloss + dloss + vloss + aloss

    def log_prob(self, true_S, mask, context=None):
        B, N = true_S.size(0), true_S.size(1)
        K = min(self.k_neighbors, N)
        mask_2D = mask.unsqueeze(1) * mask.unsqueeze(2)

        # initialize
        sloss = 0.
        S = torch.zeros(B, N).cuda().long()
        if self.context:
            h, cH, cmask, crange = self.encode_context(*context)
            X = self.predict_dist(self.O_d0(h))[0]
        else:
            X = self.init_coords(mask)[0]
            cH = cmask = None
        
        for t in range(N):
            # Predict residue t
            V, E, E_idx = self.features(X, mask)
            h = self.seq_mpn(V, E, S, E_idx, mask)
            h = self.attention(h, cH, cmask, self.W_seq)
            logits = self.O_s(h[:, t])
            prob = F.softmax(logits, dim=-1)  # [B, 20]
            snll = self.ce_loss(logits, true_S[:, t])
            sloss = sloss + torch.sum(snll * mask[:, t])

            # Teacher forcing on S
            S = S.clone()
            S[:, t] = true_S[:, t]
            S = S.clone()

            # Iterative refinement
            h = self.struct_mpn(V, E, S, E_idx, mask)
            h = self.attention(h, cH, cmask, self.W_stc)
            X = self.predict_dist(self.O_d(h))[0]

        sloss = sloss / mask.sum()
        return ReturnType(nll=sloss, X_cdr=X)

    def generate(self, B, N, context=None, return_ppl=False):
        K = min(self.k_neighbors, N)
        mask = torch.ones(B, N).cuda()

        S = torch.zeros(B, N).cuda().long()
        if self.context:
            h, cH, cmask, crange = self.encode_context(*context)
            X = self.predict_dist(self.O_d0(h))[0]
        else:
            X = self.init_coords(mask)[0]
            cH = cmask = None

        sloss = 0.
        for t in range(N):
            # Predict residue t
            V, E, E_idx = self.features(X, mask)
            h = self.seq_mpn(V, E, S, E_idx, mask)
            h = self.attention(h, cH, cmask, self.W_seq)
            logits = self.O_s(h[:, t])
            prob = F.softmax(logits, dim=-1)  # [B, 20]
            S[:, t] = torch.multinomial(prob, num_samples=1).squeeze(-1)  # [B, 1]
            sloss = sloss + self.ce_loss(logits, S[:, t])

            # Iterative refinement
            h = self.struct_mpn(V, E, S, E_idx, mask)
            h = self.attention(h, cH, cmask, self.W_stc)
            X = self.predict_dist(self.O_d(h))[0]

        S = S.tolist()
        S = [''.join([alphabet[S[i][j]] for j in range(N)]) for i in range(B)]
        ppl = torch.exp(sloss / N)
        return (S, ppl) if return_ppl else S
