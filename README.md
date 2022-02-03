# Iterative refinement graph neural network for antibody sequence-structure co-design (RefineGNN)

This is the implementation of our ICLR 2022 paper: https://arxiv.org/pdf/2110.04624.pdf

Warning: this repo is still under construction...

## Dependencies
Our model is tested in Linux with the following packages:
* CUDA >= 11.1
* PyTorch == 1.8.2 (LTS Version)
* Numpy >= 1.18.1
* tqdm

## Language model and CDR structure prediction (Section 4.1)
Our data is retreived from the Structural Antibody Database (SAbDab). The training, validation, and test data (compressed) is located in `data/sabdab`. 
To train a RefineGNN for CDR-H3, please run
```
python ab_train.py --cdr_type 3 --train_path data/sabdab/hcdr3_cluster/train.jsonl --val_path data/sabdab/hcdr3_cluster/val.jsonl --test_path data/sabdab/hcdr3_cluster/test.jsonl
```
The default hyperparameters are: hidden layer dimension `--hidden_size 256`, number of message passing layers `--depth 4`, KNN neighborhood size `--K_neighbors 9`, and the framework residue block size `--block_size 8` (multi-resolution modeling, section 3.3). The training process requires 20~24GB GPU memory. During training, this script will report perplexity (PPL) and root-mean-square-error (RMSD) over the validation set. You can also train a RefineGNN for a different CDR region by changing `--cdr_type 2` (CDR-H2) and `--cdr_type 1` (CDR-H1).

If you don't want to train RefineGNN from scratch, please load a pre-trained model and run inference on the test set by
```
python ab_train.py --cdr_type 3 --load_model ckpts/RefineGNN-hcdr3/model.best --epoch 0
```
where `--epoch 0` means zero training epochs. 

Note: GPU memory consumption can be substantially reduced by removing the multi-resolution modeling component. If you have limited GPU memory, you can train a RefineGNN without multi-resolution modeling by
```
python baseline_train.py --cdr_type 3 --train_path data/sabdab/hcdr3_cluster/train.jsonl --val_path data/sabdab/hcdr3_cluster/val.jsonl --test_path data/sabdab/hcdr3_cluster/test.jsonl --architecture RefineGNN_attonly
```
The above training script usually consumes 4GB GPU memory. You can also train our AR-GNN baseline by setting `--architecture AR-GNN`. 

## Antigen-binding antibody design (Section 4.2)
To train a RefineGNN for this task, please run
```
python ab_train.py --train_path data/rabd/train.jsonl --val_path data/rabd/val.jsonl --test_path data/rabd/test.jsonl
```
At test time, we generate 10000 CDR-H3 sequences for each antibody and select the top 100 candidates with the lowest perplexity. You can load a pre-trained model and run inference on the test set by
```
python rabd_test.py --load_model ckpts/RefineGNN-rabd/model.best
```
