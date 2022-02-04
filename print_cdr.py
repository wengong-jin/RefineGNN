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
from structgen.utils import kabsch
from tqdm import tqdm

restype_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='data/sabdab/hcdr3_cluster/test_data.jsonl')
parser.add_argument('--save_dir', default='pred_pdb/')
parser.add_argument('--load_model', required=True)
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--rmsd_threshold', type=float, default=0.8)
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

model_ckpt, opt_ckpt, model_args = torch.load(args.load_model)
model = HierarchicalDecoder(model_args).cuda()
model.load_state_dict(model_ckpt)
model.eval()

args.batch_tokens = model_args.batch_tokens
args.cdr_type = model_args.cdr_type

data = AntibodyDataset(args.data_path, cdr_type=args.cdr_type)
loader = StructureLoader(data.data, batch_tokens=args.batch_tokens, interval_sort=int(args.cdr_type))

niceprint = np.vectorize(lambda x : "%.3f" % (x,))

print("List of PDB IDs with predicted RMSD <", args.rmsd_threshold)
with torch.no_grad():
    for hbatch in tqdm(loader):
        hX, hS, hL, hmask = completize(hbatch)
        for i in range(len(hbatch)):
            pdb = hbatch[i]['pdb']
            L = hmask[i:i+1].sum().long().item()
            if L <= 3: continue

            l, r = hL[i].index(args.cdr_type), hL[i].rindex(args.cdr_type)
            N = r - l + 1
            out = model.log_prob(hS[i:i+1, :L], [hL[i]], hmask[i:i+1, :L])
            X = out.X_cdr
            rmsd = compute_rmsd(X[:, :, 1, :], hX[i:i+1, l:r+1, 1, :], hmask[i:i+1, l:r+1])  # alpha carbon

            if rmsd.item() < args.rmsd_threshold:
                _, R, t = kabsch(X[:, :, 1], hX[i:i+1, l:r+1, 1])
                X = X.view(1, N * 4, 3)
                X = torch.bmm(R, X.transpose(1,2)).transpose(1,2) + t
                X = X.view(1, N, 4, 3)
                X = X.cpu().numpy()
                print(pdb, f'RMSD={rmsd.item():.4f}')

                path = os.path.join(args.save_dir, f'{pdb}.pdb')
                with open(path, 'w') as f:
                    for j in range(N):
                        aaname = hbatch[i]['seq'][l + j]
                        aaname = restype_1to3[aaname]
                        print(f'ATOM    {j + 105}  CA  {aaname} H {j + 105}     ' + ' '.join(niceprint(X[0, j, 1, :])) + '  1.00  4.89           C', file=f)

