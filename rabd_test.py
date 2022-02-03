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
    if args.architecture == 'RefineGNN':
        return HierarchicalDecoder(args).cuda()
    elif args.architecture == 'AR-GNN':
        return Decoder(args, return_coords=False).cuda()
    elif args.architecture == 'LSTM':
        return Seq2Seq(args).cuda()
    else:
        raise ValueError('Unknown architecture')


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='data/rabd/test.jsonl')
parser.add_argument('--architecture', default='RefineGNN')
parser.add_argument('--load_model', required=True)
parser.add_argument('--seed', type=int, default=7)

args = parser.parse_args()
args.cdr_type = '3'
args.context = True

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if args.architecture == "RefineGNN":
    data = AntibodyDataset(args.data_path).data
else:
    data = CDRDataset(args.data_path, hcdr=args.cdr_type, lcdr="").cdrs

model_ckpt, opt_ckpt, model_args = torch.load(args.load_model)
model_args.architecture = args.architecture
model = build_model(model_args)
model.load_state_dict(model_ckpt)
model.eval()

succ, tot = 0, 0
with torch.no_grad():
    for ab in tqdm(data):
        new_cdrs, new_ppl = [], []
        batch_size = 5000
        for _ in range(10000 // batch_size):
            if args.architecture == 'RefineGNN':
                orig_cdr = ''.join([x for x,y in zip(ab['seq'], ab['cdr']) if y == args.cdr_type])
                hX, hS, hL, hmask = completize([ab] * batch_size)
                batch_cdrs, batch_ppl, batch_X = model.generate(hS, hL, hmask, return_ppl=True)
            else:
                orig_cdr = ab['seq']
                (hX, hS, hL, hmask), context = featurize([ab] * batch_size, context=True)
                batch_cdrs, batch_ppl = model.generate(hS.size(0), hS.size(1), context=context, return_ppl=True)

            new_cdrs.extend(batch_cdrs)
            new_ppl.extend(batch_ppl.tolist())

        new_res = sorted(zip(new_cdrs, new_ppl), key=lambda x:x[1])
        for cdr,ppl in new_res[:100]:
            match = [int(a == b) for a,b in zip(orig_cdr, cdr)]
            succ += sum(match) 
            tot += len(match)
            print(ab['pdb'], orig_cdr, cdr, ppl, match)

print(f'Amino acid recovery rate = {succ / tot:.4f}')
