import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from structgen.encoder import MPNEncoder
from structgen.data import alphabet
from structgen.utils import *
from structgen.protein_features import ProteinFeatures


class Decoder(nn.Module):
    
    def __init__(self, args, return_coords=True):
        super(Decoder, self).__init__()
        self.k_neighbors = args.k_neighbors
        self.depth = args.depth
        self.hidden_size = args.hidden_size
        self.augment_eps = args.augment_eps
        self.context = args.context
        self.return_coords = return_coords

        self.pos_embedding = PosEmbedding(16)
        self.features = ProteinFeatures(
                top_k=args.k_neighbors, num_rbf=args.num_rbf,
                features_type='dist',
                direction='forward'
        )
        self.node_in, self.edge_in = self.features.feature_dimensions['dist']
        self.O_nei = nn.Sequential(
                nn.Linear(args.hidden_size * 2 + 16, args.hidden_size),
                nn.ReLU(),
                nn.Linear(args.hidden_size, 1),
        )
        self.O_dist = nn.Sequential(
                nn.Linear(args.hidden_size * 2 + 16, args.hidden_size),
                nn.ReLU(),
                nn.Linear(args.hidden_size, 1),
        )
        self.O_s = nn.Linear(args.hidden_size, args.vocab_size)
        self.O_v = nn.Linear(args.hidden_size, self.node_in)
        self.O_e = nn.Linear(args.hidden_size, self.edge_in - self.features.num_positional_embeddings)

        self.struct_mpn = MPNEncoder(args, self.node_in, self.edge_in)
        self.seq_mpn = MPNEncoder(args, self.node_in, self.edge_in)

        if args.context:
            self.W_stc = nn.Sequential(
                    nn.Linear(args.hidden_size * 2, args.hidden_size),
                    nn.ReLU(),
            )
            self.W_seq = nn.Sequential(
                    nn.Linear(args.hidden_size * 2, args.hidden_size),
                    nn.ReLU(),
            )
            self.crnn = nn.GRU(
                    len(alphabet), args.hidden_size, 
                    batch_first=True, num_layers=1,
                    dropout=args.dropout
            )

        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    # Q: [B, N, H], K, V: [B, M, H]
    def attention(self, Q, context, W):
        context, cmask = context  # cmask: [B, M]
        att = torch.bmm(Q, context.transpose(1, 2))  # [B, N, M]
        att = att - 1e6 * (1 - cmask.unsqueeze(1))
        att = F.softmax(att, dim=-1)
        out = torch.bmm(att, context)  # [B, N, M] * [B, M, H]
        out = torch.cat([Q, out], dim=-1)
        return W(out)

    def encode_context(self, context):
        cS, cmask, crange = context
        cS = F.one_hot(cS, num_classes=len(alphabet)).float()
        cH, _ = self.crnn(cS)
        return (cH, cmask)

    def forward(self, X, S, L, mask, context=None, debug=False):
        # X: [B, N, 4, 3], S: [B, N], mask: [B, N] 
        true_V, _, _ = self.features(X, mask)
        N, K = S.size(1), self.k_neighbors

        # data augmentation
        V, E, E_idx = self.features(
                X + self.augment_eps * torch.randn_like(X), 
                mask
        )

        # run struct MPN
        h = self.struct_mpn(V, E, S, E_idx, mask)
        if self.context:
            context = self.encode_context(context)
            h = self.attention(h, context, self.W_stc)

        # predict node feature with h_{v-1}
        vout = self.O_v(h[:, :-1])
        vloss = self.mse_loss(vout, true_V[:, 1:]).mean(dim=-1)
        vloss = torch.sum(vloss * mask[:, 1:]) / mask[:, 1:].sum()

        # predict neighbors with h_{v-1}, h_u, E_pos
        E_next, nlabel, dlabel, nmask = get_nei_label(X, mask, K)  # [B, N-1, N]
        h_cur = h[:, :-1].unsqueeze(2).expand(-1,-1,N,-1)  # [B, N-1, N, H]
        h_pre = gather_nodes(h, E_next)  # [B, N-1, N, H]
        pos = torch.arange(1, N).cuda().view(1, -1, 1) - E_next  # [B, N-1, N]
        E_pos = self.pos_embedding(pos)  # [B, N-1, N, H]
        h_nei = torch.cat([h_cur, h_pre, E_pos], dim=-1)
        nout = self.O_nei(h_nei).squeeze(-1)  # [B, N-1, N]
        nloss = self.bce_loss(nout, nlabel.float())
        nloss = torch.sum(nloss * nmask) / nmask.sum()

        # predict neighbors distance
        dout = self.O_dist(h_nei).squeeze(-1)  # [B, N-1, N]
        dout = dout[:, :, :K]  # [B, N-1, K]
        dmask = nmask[:, :, :K]  # [B, N-1, K]
        dlabel = dlabel.clamp(max=20)
        dlabel = (dlabel[:, :, :K] - 10) / 10  # D in [0, 20]
        dloss = self.mse_loss(dout, dlabel)
        dloss = torch.sum(dloss * dmask) / dmask.sum()

        # sequence prediction
        h = self.seq_mpn(V, E, S, E_idx, mask)
        if self.context:
            h = self.attention(h, context, self.W_seq)

        sout = self.O_s(h)
        sloss = self.ce_loss(sout.view(-1, sout.size(-1)), S.view(-1))
        sloss = torch.sum(sloss * mask.view(-1)) / mask.sum()

        loss = sloss + nloss + vloss + dloss
        dout = dout * 10 + 10
        return (sout, vout, nout, dout) if debug else loss
    
    def expand_one_residue(self, h, V, E, E_idx, t):
        # predict node feature for t+1
        B, K = len(h), self.k_neighbors
        V[:, t+1] = self.O_v(h[:, t])

        # predict neighbors for t+1
        h_cur = h[:, t:t+1].expand(-1, t+1, -1)  # [B, t+1, H]
        h_pre = h[:, :t+1]  # [B, t+1, H]
        pos = t + 1 - torch.arange(t + 1).view(1, -1, 1).expand(B, -1, -1)  # [B, t+1, 1]
        E_pos = self.pos_embedding(pos.cuda()).squeeze(2)  # [B, t+1, H]
        h_nei = torch.cat([h_cur, h_pre, E_pos], dim=-1)
        nout = self.O_nei(h_nei).squeeze(-1)  # [B, t+1]

        if K <= t + 1:
            _, E_idx[:, t+1] = nout.topk(dim=-1, k=K, largest=True)
            nei_topk = E_idx[:, t+1]  # [B, K]
        else:
            E_idx[:, t+1, :t+1] *= 0
            E_idx[:, t+1, :t+1] += torch.arange(t, -1, -1).view(1,-1).cuda()
            nei_topk = E_idx[:, t+1, :t+1]  # [B, t+1]

        # predict neighbors distance
        # Positional encoding is relative!
        dout = self.O_dist(h_nei).squeeze(-1)  # [B, t+1]
        dout = dout * 10 + 10
        dout = gather_2d(dout, nei_topk)  # [B, t+1]
        rbf_vecs = self.features._rbf(dout.unsqueeze(1))  # [B, 1, t+1, H]
        pos_vecs = self.pos_embedding(nei_topk.unsqueeze(1) - t - 1)  # [B, 1, t+1] => [B, 1, t+1, H]
        E[:, t+1, :t+1] = torch.cat([pos_vecs, rbf_vecs], dim=-1).squeeze(1)  # [B, t+1, H]
        return nout, dout

    def log_prob(self, S, mask, context=None, debug=None):
        B, N = S.size(0), S.size(1)
        K = self.k_neighbors

        V = torch.zeros(B, N+1, self.node_in).cuda()
        V[:, :, :self.node_in // 2] = 1.  # cos(0) = 1
        E = torch.zeros(B, N+1, K, self.edge_in).cuda()
        E_idx = torch.zeros(B, N+1, K).long().cuda() + N - 1
        h_stc = [torch.zeros(B, N, self.hidden_size, requires_grad=True).cuda() for _ in range(self.depth + 1)]
        h_seq = [torch.zeros(B, N, self.hidden_size, requires_grad=True).cuda() for _ in range(self.depth + 1)]

        D = torch.zeros(B, N+1, K).cuda()
        log_prob = []
        if self.context:
            context = self.encode_context(context)

        for t in range(N):
            # run MPN
            h_seq = self.seq_mpn.inc_forward(V, E, S, E_idx, mask, h_seq, t)
            h_stc = self.struct_mpn.inc_forward(V, E, S, E_idx, mask, h_stc, t)

            h = h_seq[-1][:, t:t+1]
            if self.context:
                h = self.attention(h, context, self.W_seq)

            # predict residue for t
            logits = self.O_s(h.squeeze(1))
            lprob = F.log_softmax(logits, dim=-1)
            nll = F.nll_loss(lprob, S[:, t], reduction='none')
            log_prob.append(nll)

            # predict position for t + 1
            h = self.attention(h_stc[-1], context, self.W_stc) if self.context else h_stc[-1]
            V, E, E_idx = V.clone(), E.clone(), E_idx.clone()  # avoid inplace autograd error
            nout, dout = self.expand_one_residue(h, V, E, E_idx, t)
            V, E, E_idx = V.clone(), E.clone(), E_idx.clone()  # avoid inplace autograd error
            D[:, t+1, :dout.size(-1)] = dout

            if debug and t < N - 1:
                self.debug_decode(debug, logits, V, E, E_idx, mask, nout, dout, t)

        log_prob = torch.stack(log_prob, dim=1)  # [B, N]
        ppl = torch.sum(log_prob * mask, dim=-1) / mask.sum(dim=-1)
        log_prob = torch.sum(log_prob * mask) / mask.sum()
        if self.return_coords:
            X = fit_coords(D[:, :-1, :].detach(), E_idx[:, :-1, :].detach(), mask)
            X = X.unsqueeze(2).expand(-1,-1,4,-1)
            return ReturnType(nll=log_prob, ppl=ppl, X_cdr=X)
        else:
            return ReturnType(nll=log_prob, ppl=ppl, X_cdr=None)

    def generate(self, B, N, context=None, return_ppl=False):
        K = self.k_neighbors
        S = torch.zeros(B, N).long().cuda()
        mask = torch.ones(B, N).cuda()

        V = torch.zeros(B, N+1, self.node_in).cuda()
        V[:, :, :self.node_in // 2] = 1.  # cos(0) = 1
        E = torch.zeros(B, N+1, K, self.edge_in).cuda()
        E_idx = torch.zeros(B, N+1, K).long().cuda() + N - 1
        h_stc = [torch.zeros(B, N, self.hidden_size).cuda() for _ in range(self.depth + 1)]
        h_seq = [torch.zeros(B, N, self.hidden_size).cuda() for _ in range(self.depth + 1)]

        if self.context:
            context = self.encode_context(context)

        sloss = 0.
        for t in range(N):
            # run MPN
            h_seq = self.seq_mpn.inc_forward(V, E, S, E_idx, mask, h_seq, t)
            h_stc = self.struct_mpn.inc_forward(V, E, S, E_idx, mask, h_stc, t)

            h = h_seq[-1][:, t:t+1]
            if self.context:
                h = self.attention(h, context, self.W_seq)

            # predict residue for t
            logits = self.O_s(h.squeeze(1))
            prob = F.softmax(logits, dim=-1)  # [B, 20]
            S[:, t] = torch.multinomial(prob, num_samples=1).squeeze(-1)  # [B, 1]
            sloss = sloss + self.ce_loss(logits, S[:, t])

            # predict position for t + 1
            h = self.attention(h_stc[-1], context, self.W_stc) if self.context else h_stc[-1]
            nout, dout = self.expand_one_residue(h, V, E, E_idx, t)

        S = S.tolist()
        S = [''.join([alphabet[S[i][j]] for j in range(N)]) for i in range(B)]
        ppl = torch.exp(sloss / N)
        return (S, ppl) if return_ppl else S

    def debug_decode(self, debug_info, logits, V, E, E_idx, mask, nout, dout, t):
        X, L, true_logits, true_vout, true_nout, true_dout = debug_info[:7]
        true_V, true_E, true_E_idx = self.features(X, mask)

        print(t)
        ll = min(t + 1, self.k_neighbors)
        print('-------S-------')
        print(logits - true_logits[:, t])
        print('-------N-------')
        print(E_idx[:, t+1])
        print(true_E_idx[:, t+1])
        print(nout[:, :ll].sum() - true_nout[:, t, :ll].sum())
        print('-------V-------')
        print(V[:, t+1] - true_vout[:, t])
        print('-------E-------')
        print(dout[:, :ll].sum() - true_dout[:, t, :ll].sum())
        #print(E[:, t+1] - true_E[:, t+1])
        print('---------------')

        V[:, t+1] = true_V[:, t+1]
        E[:, t+1] = true_E[:, t+1]
        E_idx[:, t+1] = true_E_idx[:, t+1]
        input("Press Enter to continue...")
