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
from copy import deepcopy
from tqdm import tqdm, trange

from structgen import *
from neut_model import *


class CovidNeutralizationModel():
    
    def __init__(self):
        MODEL_PATH = "ckpts/covid/neut_model.ckpt"
        MODEL_ARGS = {
            "hidden_dim": 256,
            "n_layers": 2,
            "use_srupp": False
        }
        MODEL_ARGS.update(MultiABOnlyCoronavirusModel.add_extra_args())
        self.model = MultiABOnlyCoronavirusModel.load_from_checkpoint(
            MODEL_PATH,
            **MODEL_ARGS,  # type: ignore
        )
        self.model.cuda()
        self.model.eval()

    def predict(self, vh, vl):
        ab_sequence = vh
        ab_sequence = torch.tensor([AA_VOCAB[aa] for aa in ab_sequence]).long()
        ab_sequence = ab_sequence.unsqueeze(0).cuda()
        neut_logits = self.model(ab_sequence)
        prob = torch.sigmoid(neut_logits)
        return prob[0, 1].item()


def extract_fn(query):
    all_viruses = [
        x.replace("(weak)", "").strip()
        for x in query.replace(" and ", ";").replace(",", ";").split(";")
    ]
    filtered = [virus for virus in all_viruses if virus in RELEVANT_VIRUSES]
    return set(filtered)


def load_data():
    # Load antibody data
    with open("data/covabdab/CoV-AbDab_050821.csv", "r") as f:
        full_data = list(csv.DictReader(f))

    # First filter to relevant AB's
    full_data = [
        item
        for item in full_data
        if any(
            virus in item[key]
            for virus in RELEVANT_VIRUSES
            for key in RELEVANT_KEYS
        )
        and item["Ab or Nb"] == "Ab"  # remove nanobodies
        and len(item["VH or VHH"].strip()) > 2  # ensure sequence available
        and len(item["VL"].strip()) > 2  # ensure sequence available
        and "S" in item["Protein + Epitope"]  # ensure binds to S protein
        and TYPE_MAP[item["Protein + Epitope"]] == "rbd"
    ]

    sarscov2_ab_data = []
    for item in full_data:
        bindings = set()
        if item["Binds to"]:
            bindings = extract_fn(item["Binds to"])

        non_bindings = set()
        if item["Doesn't Bind to"]:
            non_bindings = extract_fn(item["Doesn't Bind to"])

        neutralizing = set()
        if item["Neutralising Vs"]:
            neutralizing = extract_fn(item["Neutralising Vs"])

        non_neutralizing = set()
        if item["Not Neutralising Vs"]:
            non_neutralizing = extract_fn(item["Not Neutralising Vs"])

        all_viruses = bindings | non_bindings | neutralizing | non_neutralizing
        full_label = [0, 0]
        full_mask = [0, 0]
        for virus in all_viruses:
            label = [0, 0]
            mask = [0, 0]
            if virus in bindings and neutralizing:
                label = [1, 1]
                mask = [1, 1]
            elif virus in bindings and non_neutralizing:
                label = [0, 1]
                mask = [1, 1]
            elif virus in non_bindings:
                label = [0, 0]
                mask = [1, 1]
            elif virus in neutralizing:
                label = [1, 1]
                mask = [1, 1]
            elif virus in bindings:
                label = [0, 1]
                mask = [0, 1]
            elif virus in non_neutralizing:
                label = [0, 1]
                mask = [1, 0]

            idx = virus == "SARS-CoV2"
            full_label[idx] = label[0]
            full_mask[idx] = mask[0]

        # Save sars-cov2 data for later
        sarscov2_ab_data.append({
            'name': item["\ufeffName"],
            'label': full_label[1],
            'mask': full_mask[1],
            'epitope': TYPE_MAP[item["Protein + Epitope"]],
            'ab': item["VH or VHH"].replace(" ", "") + "-" + item["VL"].replace(" ", ""),
            'hcdr3': item["CDRH3"],
            'lcdr3': item["CDRL3"],
        })
    return sarscov2_ab_data


def make_entry(d, args):
    entry = {k: d[k] for k in ['name', 'hcdr3', 'lcdr3']}
    entry['VH'], entry['VL'] = d['ab'].split('-')
    assert entry['VH'].count(entry['hcdr3']) == 1
    entry['context'] = entry['VH'].replace(entry['hcdr3'], '#' * len(entry['hcdr3']))
    fw1, fw2 = entry['context'].replace('#', ' ').split()
    entry['cdr'] = '0' * len(fw1) + '3' * len(entry['hcdr3']) + '0' * len(fw2)
    entry['coords'] = {
            "N": np.zeros((len(entry['VH']), 3)),
            "CA":np.zeros((len(entry['VH']), 3)),
            "C": np.zeros((len(entry['VH']), 3)),
            "O": np.zeros((len(entry['VH']), 3)),
    }
    entry['label'] = d['label']
    entry['seq'] = entry['VH'] if args.architecture == 'hierarchical' else entry['hcdr3']
    return entry


# Decode new sequences
def decode(model, ab, args):
    batch = [ab] * args.batch_size
    model.eval()
    with torch.no_grad():
        if args.architecture == 'hierarchical':
            hX, hS, hL, hmask = completize(batch)
            new_seqs = model.generate(hS, hL, hmask)
        else:
            (hX, hS, hL, hmask), context = featurize(batch, context=True)
            new_seqs = model.generate(hS.size(0), hS.size(1), context=context)
    return new_seqs


def evaluate(model, predictor, evaluator, data, args):
    succ, tot = 0, 0
    model.eval()
    with torch.no_grad():
        for ab in tqdm(data):
            new_seqs = decode(model, ab, args)
            tot = tot + len(new_seqs)
            prior_ppl = is_natural_seq(evaluator, ab, new_seqs)
            for new_cdr, ppl in zip(new_seqs, prior_ppl):
                if is_valid_seq(new_cdr) and ppl <= args.max_prior_ppl:
                    VH = ab['VH'].replace(ab['hcdr3'], new_cdr)
                    prob = predictor.predict(VH, ab['VL'])
                    succ = succ + prob
                else:
                    succ = succ + ab['score']  # not valid, improvement=0
    return succ / tot


# Glycoslation is bad
def is_valid_seq(cdr):
    if '#' in cdr: return False
    charge = [CHARGE[x] for x in cdr]
    if sum(charge) >= 3 or sum(charge) <= -3:
        return False
    for a in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        if ('N' + a + 'T') in cdr:
            return False
        if ('N' + a + 'S') in cdr:
            return False
        if (a * 4) in cdr:
            return False
    return True


def is_natural_seq(evaluator, ab, cand_cdrs):
    with torch.no_grad():
        batch = []
        for cdr in cand_cdrs:
            ab = deepcopy(ab)
            ab['seq'] = ab['VH'].replace(ab['hcdr3'], cdr)
            batch.append(ab)

        hX, hS, hL, hmask = completize(batch)
        cand_ppl1 = evaluator[0].log_prob(hS, hL, hmask).ppl.exp()

        batch = []
        for cdr in cand_cdrs:
            ab = deepcopy(ab)
            ab['seq'] = cdr
            batch.append(ab)

        (hX, hS, hL, hmask), context = featurize(batch, context=True)
        cand_ppl2 = evaluator[1].log_prob(hS, hmask, context=context).ppl.exp()
        cand_ppl3 = evaluator[2].log_prob(hS, hmask, context=context).ppl.exp()
        cand_ppl = torch.maximum(cand_ppl1, torch.maximum(cand_ppl2, cand_ppl3))

    return cand_ppl.tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster', default='data/covabdab/cdrh3_split.txt')
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--load_model', default=None)
    parser.add_argument('--log_file', default='log.txt')

    parser.add_argument('--architecture', default='hierarchical')
    parser.add_argument('--cdr_type', default='3')

    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--k_neighbors', type=int, default=9)
    parser.add_argument('--update_freq', type=int, default=1)
    parser.add_argument('--augment_eps', type=float, default=3.0)
    parser.add_argument('--block_size', type=int, default=8)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--vocab_size', type=int, default=21)
    parser.add_argument('--num_rbf', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--valid_iter', type=int, default=1000)
    parser.add_argument('--max_prior_ppl', type=float, default=10)

    args = parser.parse_args()
    args.graft = 0
    args.context = True
    print(args)

    os.makedirs(args.save_dir, exist_ok=True)
    log_file = open(os.path.join(args.save_dir, args.log_file), 'w')

    split_map = {}
    with open(args.cluster) as f:
        for line in f:
            cdr3, fold = line.strip("\r\n ").split()
            split_map[cdr3] = fold

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    predictor = CovidNeutralizationModel()
    model = HierarchicalDecoder(args).cuda()

    # Language model ensemble (ensure CDR naturalness) 
    evaluator = [None, None, None]
    eval_args = deepcopy(args)
    eval_args.hidden_size = 256
    eval_args.depth = 4
    eval_args.block_size = 8
    evaluator[0] = HierarchicalDecoder(eval_args).cuda()
    evaluator[0].eval()
    model_ckpt = torch.load("ckpts/covid/hieratt.ckpt")[0]
    evaluator[0].load_state_dict(model_ckpt)

    eval_args = deepcopy(args)
    eval_args.hidden_size = 128
    eval_args.depth = 1
    evaluator[1] = Seq2Seq(eval_args).cuda()
    evaluator[1].eval()
    model_ckpt = torch.load("ckpts/covid/lstm.ckpt")[0]
    evaluator[1].load_state_dict(model_ckpt)

    eval_args = deepcopy(args)
    eval_args.hidden_size = 256
    eval_args.depth = 3
    evaluator[2] = Decoder(eval_args, return_coords=False).cuda()
    evaluator[2].eval()
    model_ckpt = torch.load("ckpts/covid/autoreg.ckpt")[0]
    evaluator[2].load_state_dict(model_ckpt)

    optimizer = torch.optim.Adam(model.parameters())
    if args.load_model:
        model_ckpt, opt_ckpt = torch.load(args.load_model)
        model.load_state_dict(model_ckpt)
        optimizer.load_state_dict(opt_ckpt)

    all_ab = [make_entry(d, args) for d in load_data() if d['hcdr3'] in d['ab'] and d['mask'] == 1]
    for entry in tqdm(all_ab):
        entry['score'] = predictor.predict(entry['VH'], entry['VL'])

    train_ab = [d for d in all_ab if split_map[d['hcdr3']] == 'train' and d['label'] == 1]
    val_ab = [d for d in all_ab if split_map[d['hcdr3']] == 'val' and d['label'] == 1]
    test_ab = [d for d in all_ab if split_map[d['hcdr3']] == 'test' and d['label'] == 1]
    print("train/val/test:", len(train_ab), len(val_ab), len(test_ab))

    train_data = {entry['name'] : [entry] for entry in train_ab}
    best_score, best_epoch = -10.0, 0
    for e in trange(args.epochs):
        # Decode new cdrs
        ab = random.choice(train_ab)
        name = ab['name']
        new_seqs = decode(model, ab, args)
        #print(new_seqs)
        prior_ppl = is_natural_seq(evaluator, ab, new_seqs)
        for new_cdr, ppl in zip(new_seqs, prior_ppl):
            if is_valid_seq(new_cdr) and ppl <= args.max_prior_ppl:
                entry = deepcopy(ab)
                entry['VH'] = entry['VH'].replace(entry['hcdr3'], new_cdr)
                prob = predictor.predict(entry['VH'], entry['VL'])
                #print(name, entry['hcdr3'], entry['score'], '-->', new_cdr, prob)
                if prob > entry['score']:
                    entry['score'] = prob
                    entry['hcdr3'] = new_cdr
                    entry['seq'] = entry['VH'] if args.architecture == 'hierarchical' else entry['hcdr3']
                    train_data[name].append(entry)

        if name in train_data:
            dlist = sorted(train_data[name], key=lambda d:d['score'], reverse=True)
            train_data[name] = dlist[:args.batch_size]

        # Train model
        model.train()
        optimizer.zero_grad()
        train_keys = sorted(train_data.keys())
        name = random.choice(train_keys)
        batch = train_data[name]

        if args.architecture == 'hierarchical':
            hX, hS, hL, hmask = completize(batch)
            loss = model.log_prob(hS, hL, hmask).nll
        else:
            (hX, hS, hL, hmask), context = featurize(batch, context=True)
            loss = model.log_prob(hS, hmask, context=context).nll

        loss.backward()
        optimizer.step()

        if (e + 1) % args.valid_iter == 0:
            ckpt = (model.state_dict(), optimizer.state_dict())
            torch.save(ckpt, os.path.join(args.save_dir, f"model.ckpt.{e+1}"))
            val_score = evaluate(model, predictor, evaluator, val_ab, args)
            print(f'Epoch {e+1}: average neutralization score: {val_score:.3f}')
            if val_score > best_score:
                best_epoch = e + 1
                best_score = val_score

best_epoch = 10000
if best_epoch > 0:
    best_ckpt = os.path.join(args.save_dir, f"model.ckpt.{best_epoch}")
    model.load_state_dict(torch.load(best_ckpt)[0])

test_score = evaluate(model, predictor, evaluator, test_ab, args)
print(f'Test average neutralization score: {test_score:.3f}')
