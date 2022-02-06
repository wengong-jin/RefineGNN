import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import csv
import math, random, sys
import numpy as np
import argparse
import os

from structgen.protein_features import ProteinFeatures
from structgen.utils import compute_rmsd, self_square_dist, gather_nodes, kabsch
from collections import namedtuple
from tqdm import tqdm

torch.set_num_threads(8)

ALPHABET = ['#', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
ReturnType = namedtuple('ReturnType',('loss','bind_X'), defaults=(None, None))


class AntibodyComplexDataset():

    def __init__(self, jsonl_file, cdr_type, L_binder, L_target, language_model=True):
        self.data = []
        with open(jsonl_file) as f:
            all_lines = f.readlines()
            for line in tqdm(all_lines):
                entry = json.loads(line)
                assert len(entry['antibody_coords']) == len(entry['antibody_seq'])
                assert len(entry['antigen_coords']) == len(entry['antigen_seq'])

                # Create scaffold
                if language_model:
                    entry['scaffold_seq'] = ''.join([
                        ('#' if y in cdr_type else x) for x,y in zip(entry['antibody_seq'], entry['antibody_cdr'])
                    ])[:L_binder]
                else:
                    entry['scaffold_seq'] = entry['antibody_seq'][:L_binder]

                entry['scaffold_coords'] = torch.tensor(entry['antibody_coords'])[:L_binder]
                entry['scaffold_atypes'] = torch.tensor(entry['antibody_atypes'])[:L_binder]

                # Binding region
                entry['antibody_cdr'] = entry['antibody_cdr'][:L_binder]
                surface = torch.tensor(
                        [i for i,v in enumerate(entry['antibody_cdr']) if v in cdr_type]
                )
                entry['binder_surface'] = surface
                entry['binder_seq'] = ''.join([entry['antibody_seq'][i] for i in surface.tolist()])
                entry['binder_coords'] = entry['scaffold_coords'][surface]
                entry['binder_atypes'] = entry['scaffold_atypes'][surface]

                # Create target
                entry['target_seq'] = entry['antigen_seq']
                entry['target_coords'] = torch.tensor(entry['antigen_coords'])
                entry['target_atypes'] = torch.tensor(entry['antigen_atypes'])

                # Find target surface
                bind_X = entry['binder_coords'][:, 1]
                tgt_X = entry['target_coords'][:, 1]
                dist = bind_X[None,:,:] - tgt_X[:,None,:]  # [1, N, 3] - [M, 1, 3]
                dist = dist.norm(dim=-1, p=2).amin(dim=-1) # [M, N] -> [M]
                _, target = dist.topk(k=min(len(dist),L_target), largest=False)
                entry['target_surface'] = target

                if len(entry['binder_coords']) > 4 and len(entry['target_coords']) > 4:
                    self.data.append(entry)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ComplexLoader():

    def __init__(self, dataset, batch_tokens):
        self.dataset = dataset
        self.size = len(dataset)
        self.lengths = [len(dataset[i]['binder_seq']) for i in range(self.size)]
        self.batch_tokens = batch_tokens
        sorted_ix = np.argsort(self.lengths)

        # Cluster into batches of similar sizes
        clusters, batch = [], []
        for ix in sorted_ix:
            size = self.lengths[ix]
            batch.append(ix)
            if size * (len(batch) + 1) > self.batch_tokens:
                clusters.append(batch)
                batch = []

        self.clusters = clusters
        if len(batch) > 0:
            clusters.append(batch)

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            batch = [self.dataset[i] for i in b_idx]
            yield batch


def featurize(batch, name):
    B = len(batch)
    L_max = max([len(b[name + "_seq"]) for b in batch])
    X = torch.zeros([B, L_max, 14, 3])
    S = torch.zeros([B, L_max]).long()
    A = torch.zeros([B, L_max, 14]).long()

    # Build the batch
    for i, b in enumerate(batch):
        l = len(b[name + '_seq'])
        X[i,:l] = b[name + '_coords']
        A[i,:l] = b[name + '_atypes']
        indices = torch.tensor([ALPHABET.index(a) for a in b[name + '_seq']])
        S[i,:l] = indices

    return X.cuda(), S.cuda(), A.cuda()


def make_batch(batch):
    target = featurize(batch, 'target')
    scaffold = featurize(batch, 'scaffold')
    binder = featurize(batch, 'binder')
    surface = ([b['binder_surface'] for b in batch], [b['target_surface'] for b in batch])
    return binder, scaffold, target, surface


class MPNNLayer(nn.Module):

    def __init__(self, num_hidden, num_in, dropout):
        super(MPNNLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.dropout = nn.Dropout(dropout)
        self.W = nn.Sequential(
                nn.Linear(num_hidden + num_in, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, num_hidden),
        )

    def forward(self, h_V, h_E, mask_attend):
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E], dim=-1)  # [B, N, K, H]
        h_message = self.W(h_EV) * mask_attend.unsqueeze(-1)
        dh = torch.mean(h_message, dim=-2)
        h_V = h_V + self.dropout(dh)
        return h_V


class MPNEncoder(nn.Module):
    
    def __init__(self, args):
        super(MPNEncoder, self).__init__()
        self.features = ProteinFeatures(
                top_k=args.k_neighbors, num_rbf=args.num_rbf,
                features_type='full',
                direction='bidirectional'
        )
        self.node_in, self.edge_in = self.features.feature_dimensions['full']
        
        self.W_v = nn.Linear(self.node_in, args.hidden_size)
        self.W_e = nn.Linear(self.edge_in, args.hidden_size)
        self.layers = nn.ModuleList([
                MPNNLayer(args.hidden_size, args.hidden_size * 3, dropout=args.dropout)
                for _ in range(args.depth)
        ])
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, X, V, S, A):
        mask = A.clamp(max=1).float()
        vmask = mask[:,:,1]
        _, E, E_idx = self.features(X, vmask)

        h = self.W_v(V)    # [B, N, H] 
        h_e = self.W_e(E)  # [B, N, K, H] 
        nei_s = gather_nodes(S, E_idx)  # [B, N, K, H]
        emask = gather_nodes(vmask[...,None], E_idx).squeeze(-1)

        # message passing
        for layer in self.layers:
            nei_v = gather_nodes(h, E_idx)  # [B, N, K, H]
            nei_h = torch.cat([nei_v, nei_s, h_e], dim=-1)
            h = layer(h, nei_h, mask_attend=emask)  # [B, N, H]
            h = h * vmask.unsqueeze(-1)  # [B, N, H]
        return h


class RefineFolder(nn.Module):

    def __init__(self, args):
        super(RefineFolder, self).__init__()
        self.rstep = args.rstep
        self.k_neighbors = args.k_neighbors
        self.hidden_size = args.hidden_size
        self.embedding = nn.Embedding(len(ALPHABET), args.hidden_size)
        self.rnn = nn.GRU(
                args.hidden_size,
                args.hidden_size,
                num_layers=1,
                batch_first=True,
                dropout=args.dropout,
        )
        self.features = ProteinFeatures(
                top_k=args.k_neighbors, num_rbf=args.num_rbf,
                features_type='full',
                direction='bidirectional'
        )
        self.W_x0 = nn.Linear(args.hidden_size, 42)
        self.W_x = nn.Linear(args.hidden_size, 42)
        self.struct_mpn = MPNEncoder(args)
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        self.huber_loss = nn.SmoothL1Loss(reduction='none')

    def encode_scaffold(self, h_S, mask, bind_pos):
        scaf_h, _ = self.rnn(h_S)
        max_len = max([len(pos) for pos in bind_pos])
        bind_h = [scaf_h[i, pos] for i,pos in enumerate(bind_pos)]
        bind_h = [F.pad(h, (0,0,0,max_len-len(h))) for h in bind_h]
        return torch.stack(bind_h, dim=0), scaf_h

    def struct_loss(self, X, mask, true_D, true_V, true_AD):
        D, _ = self_square_dist(X, mask[:,:,1])
        V = self.features._dihedrals(X)
        AD = self.features._AD_features(X[:,:,1,:])
        dloss = self.huber_loss(D, true_D) + 20 * F.relu(14.4 - D)
        vloss = self.mse_loss(V, true_V).sum(dim=-1)
        aloss = self.mse_loss(AD, true_AD).sum(dim=-1)
        return dloss, vloss + aloss

    def forward(self, binder, scaffold, surface):
        true_X, true_S, true_A = binder
        _, scaf_S, scaf_A = scaffold
        surface, _ = surface
        true_mask = true_A.clamp(max=1).float()

        # Ground truth 
        B, N, L = true_X.size(0), true_X.size(1), true_X.size(2)
        true_V = self.features._dihedrals(true_X)
        true_D, mask_2D = self_square_dist(true_X, true_mask[:,:,1])
        true_AD = self.features._AD_features(true_X[:,:,1,:])

        # Initial coords
        scaf_S = self.embedding(scaf_S)
        scaf_mask = scaf_A[:,:,1].clamp(max=1).float()
        scaf_h, _ = self.encode_scaffold(scaf_S, scaf_mask, surface)

        X = self.W_x0(scaf_h).view(B, N, L, 3)
        dloss, vloss = self.struct_loss(X, true_mask, true_D, true_V, true_AD)

        for t in range(self.rstep):
            X = X.detach().clone()
            V = self.features._dihedrals(X)
            h = self.struct_mpn(X, V, scaf_h, true_A)
            X = self.W_x(h).view(B, N, L, 3)
            X = X * true_mask[...,None]
            dloss_t, vloss_t = self.struct_loss(X, true_mask, true_D, true_V, true_AD)
            dloss += dloss_t
            vloss += vloss_t

        dloss = torch.sum(dloss * mask_2D) / mask_2D.sum()
        vloss = torch.sum(vloss * true_mask[:,:,1]) / true_mask[:,:,1].sum()
        loss = dloss + vloss
        return ReturnType(loss=loss, bind_X=X.detach())


def evaluate(model, loader, args):
    model.eval()
    bb_rmsd = []
    with torch.no_grad():
        for batch in tqdm(loader):
            binder, scaffold, target, surface = make_batch(batch)[:4]
            true_X, _, true_A = binder
            true_mask = true_A.clamp(max=1).float()
            out = model(binder, scaffold, surface)
            rmsd = compute_rmsd(
                    out.bind_X[:, :, 1], true_X[:, :, 1], true_mask[:, :, 1]
            )
            bb_rmsd.extend(rmsd.tolist())

    return sum(bb_rmsd) / len(bb_rmsd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='data/sabdab_2022_01/train_data.jsonl')
    parser.add_argument('--val_path', default='data/sabdab_2022_01/val_data.jsonl')
    parser.add_argument('--test_path', default='data/sabdab_2022_01/test_data.jsonl')
    parser.add_argument('--save_dir', default='ckpts/tmp')
    parser.add_argument('--load_model', default=None)

    parser.add_argument('--cdr', default='123')

    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--batch_tokens', type=int, default=200)
    parser.add_argument('--k_neighbors', type=int, default=9)
    parser.add_argument('--L_binder', type=int, default=150)
    parser.add_argument('--L_target', type=int, default=200)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--rstep', type=int, default=4)
    parser.add_argument('--vocab_size', type=int, default=21)
    parser.add_argument('--num_rbf', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--print_iter', type=int, default=50)
    parser.add_argument('--anneal_rate', type=float, default=0.9)
    parser.add_argument('--clip_norm', type=float, default=1.0)

    args = parser.parse_args()
    print(args)

    os.makedirs(args.save_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    all_data = []
    for path in [args.train_path, args.val_path, args.test_path]:
        data = AntibodyComplexDataset(
                path,
                cdr_type=args.cdr,
                L_binder=args.L_binder,
                L_target=args.L_target,
                language_model=False
        )
        all_data.append(data)

    loader_train = ComplexLoader(all_data[0], batch_tokens=args.batch_tokens)
    loader_val = ComplexLoader(all_data[1], batch_tokens=0)
    loader_test = ComplexLoader(all_data[2], batch_tokens=0)

    model = RefineFolder(args).cuda()
    optimizer = torch.optim.Adam(model.parameters())

    if args.load_model:
        model_ckpt, opt_ckpt, model_args = torch.load(args.load_model)
        model = RefineFolder(model_args).cuda()  # new argument
        optimizer = torch.optim.Adam(model.parameters())
        model.load_state_dict(model_ckpt)
        optimizer.load_state_dict(opt_ckpt)

    print('Training:{}, Validation:{}, Test:{}'.format(
        len(loader_train.dataset), len(loader_val.dataset), len(loader_test.dataset))
    )

    best_rmsd, best_epoch = 100, -1
    for e in range(args.epochs):
        model.train()
        meter = 0

        for i,batch in enumerate(tqdm(loader_train)):
            optimizer.zero_grad()
            binder, scaffold, target, surface = make_batch(batch)[:4]
            out = model(binder, scaffold, surface)
            out.loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

            meter += out.loss.item()
            if (i + 1) % args.print_iter == 0:
                meter /= args.print_iter
                print(f'[{i + 1}] Train Loss = {meter:.3f}')
                meter = 0

        val_rmsd = evaluate(model, loader_val, args)
        ckpt = (model.state_dict(), optimizer.state_dict(), args)
        torch.save(ckpt, os.path.join(args.save_dir, f"model.ckpt.{e}"))
        print(f'Epoch {e}, Backbone RMSD = {val_rmsd:.3f}')

        if val_rmsd < best_rmsd:
            best_rmsd = val_rmsd
            best_epoch = e

    if best_epoch >= 0:
        best_ckpt = os.path.join(args.save_dir, f"model.ckpt.{best_epoch}")
        model.load_state_dict(torch.load(best_ckpt)[0])

    test_rmsd = evaluate(model, loader_test, args)
    print(f'Test Backbone RMSD = {test_rmsd:.3f}')
