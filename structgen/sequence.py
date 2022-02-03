import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from structgen.data import alphabet
from structgen.utils import ReturnType


class SeqModel(nn.Module):

    def __init__(self, args):
        super(SeqModel, self).__init__()
        self.hidden_size = args.hidden_size
        self.W_s = nn.Embedding(args.vocab_size, args.hidden_size)

        self.lstm = nn.LSTM(
                args.hidden_size, args.hidden_size, 
                batch_first=True, num_layers=args.depth,
                dropout=args.dropout
        )
        self.W_out = nn.Linear(args.hidden_size, args.vocab_size, bias=True)
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, S, mask):
        h_S = self.W_s(S)
        h_S_shift = F.pad(h_S[:,0:-1], (0,0,1,0), 'constant', 0)
        h_V, _ = self.lstm(h_S_shift)
        logits = self.W_out(h_V)
        loss = self.ce_loss(logits.view(-1, logits.size(-1)), S.view(-1))
        loss = torch.sum(loss * mask.view(-1)) / mask.sum()
        return loss

    def log_prob(self, S, mask):
        return ReturnType(nll=self(S, mask))

    def generate(self, B, N):
        h = torch.zeros(self.lstm.num_layers, B, self.hidden_size).cuda()
        c = torch.zeros(self.lstm.num_layers, B, self.hidden_size).cuda()
        S = torch.zeros(B, N + 1).long().cuda()
        for t in range(N):
            h_S = self.W_s(S[:, t:t+1])
            h_V, (h, c) = self.lstm(h_S, (h, c))
            logits = self.W_out(h_V)
            prob = F.softmax(logits, dim=-1).squeeze(1)
            S[:, t+1] = torch.multinomial(prob, num_samples=1).squeeze(-1)
        
        S = S[:, 1:].tolist()
        S = [''.join([alphabet[S[i][j]] for j in range(N)]) for i in range(B)]
        return S


class Seq2Seq(nn.Module):

    def __init__(self, args):
        super(Seq2Seq, self).__init__()
        self.hidden_size = args.hidden_size
        self.W_s = nn.Embedding(args.vocab_size, args.hidden_size)
        self.W_a = nn.Embedding(args.vocab_size, args.hidden_size)

        self.encoder = nn.LSTM(
                args.hidden_size, args.hidden_size, 
                batch_first=True, num_layers=args.depth,
                dropout=args.dropout
        )
        self.decoder = nn.LSTM(
                args.hidden_size, args.hidden_size, 
                batch_first=True, num_layers=args.depth,
                dropout=args.dropout
        )
        self.W_out = nn.Linear(args.hidden_size * 2, args.vocab_size, bias=True)
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def RL_parameters(self):
        return self.parameters()

    def encode(self, aS, amask):
        h_S = self.W_a(aS)
        h_V, _ = self.encoder(h_S)  # [B, M, H]
        return h_V * amask.unsqueeze(-1)

    def forward(self, S, mask, context):
        h_S = self.W_s(S)
        h_S_shift = F.pad(h_S[:,0:-1], (0,0,1,0), 'constant', 0)
        h_V, _ = self.decoder(h_S_shift)  # [B, N, H]

        # attention
        aS, amask, _ = context
        h_A = self.encode(aS, amask)  # [B, M, H]
        att = torch.bmm(h_V, h_A.transpose(1, 2))  # [B, N, M]
        att_mask = mask.unsqueeze(2) * amask.unsqueeze(1)  # [B, N, 1] * [B, 1, M]
        att = att - (1 - att_mask) * 1e6  # attention mask
        att = F.softmax(att, dim=-1)  # [B, N, M]
        h_att = torch.bmm(att, h_A)
        h_out = torch.cat([h_V, h_att], dim=-1)

        logits = self.W_out(h_out)
        loss = self.ce_loss(logits.view(-1, logits.size(-1)), S.view(-1))
        loss = torch.sum(loss * mask.view(-1)) / mask.sum()
        return loss

    def log_prob(self, S, mask, context):
        B, N = S.size(0), S.size(1)
        h_S = self.W_s(S)
        h_S_shift = F.pad(h_S[:,0:-1], (0,0,1,0), 'constant', 0)
        h_V, _ = self.decoder(h_S_shift)  # [B, N, H]

        # attention
        aS, amask, _ = context
        h_A = self.encode(aS, amask)  # [B, M, H]
        att = torch.bmm(h_V, h_A.transpose(1, 2))  # [B, N, M]
        att_mask = mask.unsqueeze(2) * amask.unsqueeze(1)  # [B, N, 1] * [B, 1, M]
        att = att - (1 - att_mask) * 1e6  # attention mask
        att = F.softmax(att, dim=-1)  # [B, N, M]
        h_att = torch.bmm(att, h_A)
        h_out = torch.cat([h_V, h_att], dim=-1)

        logits = self.W_out(h_out)
        loss = self.ce_loss(logits.view(-1, logits.size(-1)), S.view(-1))
        loss = loss.view(B, N)
        ppl = torch.sum(loss * mask, dim=-1) / mask.sum(dim=-1)
        nll = torch.sum(loss * mask) / mask.sum()

        return ReturnType(nll=nll, ppl=ppl)

    def generate(self, B, N, context, return_ppl=False):
        aS, amask, _ = context
        h_A = self.encode(aS, amask)  # [B, M, H]

        h = torch.zeros(self.decoder.num_layers, B, self.hidden_size).cuda()
        c = torch.zeros(self.decoder.num_layers, B, self.hidden_size).cuda()
        S = torch.zeros(B, N + 1).long().cuda()
        sloss = 0.

        for t in range(N):
            h_S = self.W_s(S[:, t:t+1])
            h_V, (h, c) = self.decoder(h_S, (h, c))

            att = torch.bmm(h_V, h_A.transpose(1, 2))  # [B, 1, M]
            att_mask = amask.unsqueeze(1)  # [B, 1, M]
            att = att - (1 - att_mask) * 1e6  # attention mask
            att = F.softmax(att, dim=-1)  # [B, 1, M]
            h_att = torch.bmm(att, h_A)   # [B, 1, H]
            h_out = torch.cat([h_V, h_att], dim=-1)

            logits = self.W_out(h_out).squeeze(1)
            prob = F.softmax(logits, dim=-1)
            S[:, t+1] = torch.multinomial(prob, num_samples=1).squeeze(-1)
            sloss = sloss + self.ce_loss(logits, S[:, t+1])
        
        S = S[:, 1:].tolist()
        S = [''.join([alphabet[S[i][j]] for j in range(N)]) for i in range(B)]
        ppl = torch.exp(sloss / N)
        return (S, ppl) if return_ppl else S


