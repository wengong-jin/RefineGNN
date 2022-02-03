# Iterative refinement graph neural network for antibody sequence-structure co-design (RefineGNN)

This is the implementation of our ICLR 2022 paper: https://arxiv.org/pdf/2110.04624.pdf

Warning: this repo is still under construction...

## Language model and CDR structure prediction (Section 4.1)
Antibody structure data is retreived from the Structural Antibody Database (SAbDab). The training, validation, and test sets are provided in `data/sabdab`. Please decompress the files in that folder. To train a generative model for CDR-H3, please run
```
python ab_train.py --cdr_type 3 
```

## Antigen-binding antibody design (Section 4.2)
antibody-antigen binding data is provided in `data/rabd`. To train a generative model, please run
```
python ab_train.py --train_path data/rabd/train.jsonl --val_path data/rabd/val.jsonl --test_path data/rabd/test.jsonl
```
