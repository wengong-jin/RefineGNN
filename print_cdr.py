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

from fold_train import *
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
parser.add_argument('--data_path', default='data/sabdab_2022_01/test_data.jsonl')
parser.add_argument('--save_dir', default='pred_pdb/')
parser.add_argument('--load_model', required=True)
parser.add_argument('--seed', type=int, default=7)
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

model_ckpt, opt_ckpt, model_args = torch.load(args.load_model)
model = RefineFolder(model_args).cuda()
model.load_state_dict(model_ckpt)
model.eval()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

data = AntibodyComplexDataset(
        args.data_path,
        cdr_type=model_args.cdr,
        L_binder=model_args.L_binder,
        L_target=model_args.L_target,
        language_model=False
)
loader = ComplexLoader(data, batch_tokens=0)

niceprint = np.vectorize(lambda x : "%.3f" % (x,))

with torch.no_grad():
    for data in tqdm(loader):
        binder, scaffold, target, surface = make_batch(data)[:4]
        binder_surface = surface[0][0].tolist()
        out = model(binder, scaffold, surface)

        bind_X, _, bind_A = binder
        bind_mask = bind_A.clamp(max=1).float()

        bb_rmsd = compute_rmsd(
                out.bind_X[:, :, 1], bind_X[:, :, 1], bind_mask[:, :, 1]
        ).item()

        pdb = data[0]['pdb']
        X = out.bind_X + 200
        X = X.cpu().numpy()

        path = os.path.join(args.save_dir, f'{pdb}.pdb')
        with open(path, 'w') as f:
            print(f'REMARK  RMSD={bb_rmsd:.4f}', file=f)
            for i in range(bind_X.size(1)):
                idx = binder_surface[i]
                aaname = data[0]['antibody_seq'][idx]
                aaname = restype_1to3[aaname]
                print(f'ATOM    924  CA  {aaname} H ' + str(binder_surface[i]) + '     ' + ' '.join(niceprint(X[0, i, 1, :])), file=f)

