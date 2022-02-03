import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import json
import csv
import math, random, sys
import numpy as np
import argparse
import os

from structgen import *
from tqdm import tqdm


def build_model(args):
    if args.architecture == 'RefineGNN_attonly':
        return RevisionDecoder(args).cuda()
    elif args.architecture == 'AR-GNN':
        return Decoder(args).cuda()
    else:
        raise ValueError('Unknown architecture')


def evaluate(model, loader, args):
    model.eval()
    val_nll = val_tot = 0.
    val_rmsd = []
    with torch.no_grad():
        for hbatch, abatch in tqdm(loader):
            (hX, hS, hL, hmask), context = featurize(hbatch)
            for i in range(len(hbatch)):
                L = hmask[i:i+1].sum().long().item()
                if L > 0:
                    context_i = (context[0][i:i+1], context[1][i:i+1], context[2][i:i+1])
                    out = model.log_prob(hS[i:i+1, :L], hmask[i:i+1, :L], context=context_i)
                    nll, X_pred = out.nll, out.X_cdr
                    val_nll += nll.item() * L if torch.isnan(nll).sum().item() == 0 else 3 * L
                    val_tot += L
                    rmsd = compute_rmsd(X_pred[:, :L, 1, :], hX[i:i+1, :L, 1, :], hmask[i:i+1, :L])  # alpha carbon
                    val_rmsd.append(rmsd.item())

    return math.exp(val_nll / val_tot), sum(val_rmsd) / len(val_rmsd)


parser = argparse.ArgumentParser()
parser.add_argument('--train_path', default='data/sabdab/hcdr3_cluster/train_data.jsonl')
parser.add_argument('--val_path', default='data/sabdab/hcdr3_cluster/val_data.jsonl')
parser.add_argument('--test_path', default='data/sabdab/hcdr3_cluster/test_data.jsonl')
parser.add_argument('--save_dir', default='ckpts/tmp')
parser.add_argument('--load_model', default=None)

parser.add_argument('--hcdr', default='3')
parser.add_argument('--architecture', default='RefineGNN_attonly')

parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--batch_tokens', type=int, default=100)
parser.add_argument('--k_neighbors', type=int, default=9)
parser.add_argument('--augment_eps', type=float, default=3.0)
parser.add_argument('--depth', type=int, default=4)
parser.add_argument('--vocab_size', type=int, default=21)
parser.add_argument('--num_rbf', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.1)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--clip_norm', type=float, default=5.0)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--anneal_rate', type=float, default=0.9)
parser.add_argument('--print_iter', type=int, default=50)

args = parser.parse_args()
args.context = True
print(args)

os.makedirs(args.save_dir, exist_ok=True)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

loaders = []
for path in [args.train_path, args.val_path, args.test_path]:
    data = CDRDataset(path, hcdr=args.hcdr)
    loader = StructureLoader(data.cdrs, batch_tokens=args.batch_tokens, binder_data=data.atgs)
    loaders.append(loader)

loader_train, loader_val, loader_test = loaders
model = build_model(args)
optimizer = torch.optim.Adam(model.parameters())

if args.load_model:
    model_ckpt, opt_ckpt, model_args = torch.load(args.load_model)
    model = build_model(model_args)
    model.load_state_dict(model_ckpt)
    optimizer.load_state_dict(opt_ckpt)

print('Training:{}, Validation:{}, Test:{}'.format(
    len(loader_train.dataset), len(loader_val.dataset), len(loader_test.dataset))
)

best_ppl, best_epoch = 100, -1
for e in range(args.epochs):
    model.train()
    meter = 0

    for i, (hbatch,abatch) in enumerate(tqdm(loader_train)):
        optimizer.zero_grad()
        hchain, context = featurize(hbatch)
        if hchain[-1].sum().item() == 0:
            continue

        loss = model(*hchain, context=context)
        loss.backward()
        optimizer.step()

        meter += loss.item()
        if (i + 1) % args.print_iter == 0:
            meter /= args.print_iter
            print(f'[{i + 1}] Train Loss = {meter:.3f}')
            meter = 0

    val_ppl, val_rmsd = evaluate(model, loader_val, args)
    ckpt = (model.state_dict(), optimizer.state_dict())
    torch.save(ckpt, os.path.join(args.save_dir, f"model.ckpt.{e}"))
    print(f'Epoch {e}, Val PPL = {val_ppl:.3f}, Val RMSD = {val_rmsd:.3f}')

    if val_ppl < best_ppl:
        best_ppl = val_ppl
        best_epoch = e

if best_epoch >= 0:
    best_ckpt = os.path.join(args.save_dir, f"model.ckpt.{best_epoch}")
    model.load_state_dict(torch.load(best_ckpt)[0])

test_ppl, test_rmsd = evaluate(model, loader_test, args)
print(f'Test PPL = {test_ppl:.3f}, Test RMSD = {test_rmsd:.3f}')
