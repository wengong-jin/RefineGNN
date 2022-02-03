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
from collections import namedtuple

def evaluate_ppl(model, loader, args):
    model.eval()
    with torch.no_grad():
        val_nll = val_tot = 0.
        for hbatch, _ in tqdm(loader):
            (hX, hS, hL, hmask), context = featurize(hbatch)
            nll = model(hS, hmask, context) * hmask.sum()
            val_nll += nll.item()
            val_tot += hmask.sum().item()
    return math.exp(val_nll / val_tot)


parser = argparse.ArgumentParser()
parser.add_argument('--train_path', default='data/sabdab/hcdr3_cluster/train_data.jsonl')
parser.add_argument('--val_path', default='data/sabdab/hcdr3_cluster/val_data.jsonl')
parser.add_argument('--test_path', default='data/sabdab/hcdr3_cluster/test_data.jsonl')
parser.add_argument('--save_dir', required=True)
parser.add_argument('--load_model', default=None)

parser.add_argument('--hcdr', default='3')
parser.add_argument('--lcdr', default='')

parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--batch_tokens', type=int, default=400)
parser.add_argument('--depth', type=int, default=2)
parser.add_argument('--vocab_size', type=int, default=21)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--clip_norm', type=float, default=5.0)

parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--anneal_rate', type=float, default=0.9)
parser.add_argument('--print_iter', type=int, default=50)

args = parser.parse_args()
print(args)

os.makedirs(args.save_dir, exist_ok=True)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

loaders = []
for path in [args.train_path, args.val_path, args.test_path]:
    data = CDRDataset(path, hcdr=args.hcdr, lcdr=args.lcdr)
    loader = StructureLoader(data.cdrs, batch_tokens=args.batch_tokens, binder_data=data.atgs)
    loaders.append(loader)

loader_train, loader_val, loader_test = loaders
model = Seq2Seq(args).cuda()
optimizer = torch.optim.Adam(model.parameters())

if args.load_model:
    model_ckpt, opt_ckpt, model_args = torch.load(args.load_model)
    model.load_state_dict(model_ckpt)
    optimizer.load_state_dict(opt_ckpt)

print('Training:{}, Validation:{}, Test:{}'.format(
    len(loader_train.dataset), len(loader_val.dataset), len(loader_test.dataset))
)

best_ppl, best_epoch = 100, None
for e in range(args.epochs):
    model.train()
    meter = 0
    for i, (hbatch, _) in enumerate(loader_train):
        optimizer.zero_grad()
        (hX, hS, hL, hmask), context = featurize(hbatch)
        if hmask.sum().item() == 0:
            continue
        loss = model(hS, hmask, context)
        loss.backward()
        optimizer.step()

        meter += torch.exp(loss).item()
        if (i + 1) % args.print_iter == 0:
            meter /= args.print_iter
            print(f'[{i + 1}] Train PPL = {meter:.3f}')
            meter = 0

    val_ppl = evaluate_ppl(model, loader_val, args)
    ckpt = (model.state_dict(), optimizer.state_dict())
    torch.save(ckpt, os.path.join(args.save_dir, f"model.ckpt.{e}"))
    print(f'Epoch {e}, Val PPL = {val_ppl:.3f}')

    if val_ppl < best_ppl:
        best_ppl = val_ppl
        best_epoch = e

if best_epoch:
    best_ckpt = os.path.join(args.save_dir, f"model.ckpt.{best_epoch}")
    model.load_state_dict(torch.load(best_ckpt)[0])

test_ppl = evaluate_ppl(model, loader_test, args)
print(f'Test PPL = {test_ppl:.3f}')
