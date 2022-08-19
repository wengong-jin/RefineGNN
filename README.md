# Iterative refinement graph neural network for antibody sequence-structure co-design (RefineGNN)

This is the implementation of our ICLR 2022 paper: https://arxiv.org/pdf/2110.04624.pdf

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
python ab_train.py --cdr_type 3 --train_path data/sabdab/hcdr3_cluster/train_data.jsonl --val_path data/sabdab/hcdr3_cluster/val_data.jsonl --test_path data/sabdab/hcdr3_cluster/test_data.jsonl
```
The default hyperparameters are: hidden layer dimension `--hidden_size 256`, number of message passing layers `--depth 4`, KNN neighborhood size `--K_neighbors 9`, and the framework residue block size `--block_size 8` (multi-resolution modeling, section 3.3).

During training, this script will report perplexity (PPL) and root-mean-square-error (RMSD) over the validation set. You can also train a RefineGNN for a different CDR region by changing `--cdr_type 2` (CDR-H2) and `--cdr_type 1` (CDR-H1).

If you don't want to train RefineGNN from scratch, please load a pre-trained model and run inference on the test set by
```
python ab_train.py --cdr_type 3 --load_model ckpts/RefineGNN-hcdr3/model.best --epoch 0
```
where `--epoch 0` means zero training epochs. 

Note: The above training script usually requires 20~24GB GPU memory. The GPU memory consumption can be substantially reduced by removing the multi-resolution modeling component. If you have limited GPU memory, you can train a RefineGNN without multi-resolution modeling by
```
python baseline_train.py --cdr_type 3 --train_path data/sabdab/hcdr3_cluster/train_data.jsonl --val_path data/sabdab/hcdr3_cluster/val_data.jsonl --test_path data/sabdab/hcdr3_cluster/test_data.jsonl --architecture RefineGNN_attonly
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

## CDR Structure Prediction
Besides CDR sequence design, we can also use RefineGNN to predict CDR loop structure given an antibody VH sequence. To train a RefineGNN for structure prediction alone, please run
```
python fold_train.py --cdr 123
```
`--cdr 123` means the model will predict CDR-H1, CDR-H2, and CDR-H3 combined. You can change it to `--cdr 3` if you want to predict CDR-H3 structure only.

For convenience, we have provided a pre-trained checkpoint for CDR-H1,2,3 joint structure prediction. You can print the predicted CDR structures using the following script:
```
python print_cdr.py --load_model ckpts/RefineGNN-hfold/model.best --save_dir pred_pdbs
```
The predicted structures are saved in `pred_pdbs/*.pdb`. Each pdb file has a header line that reports the RMSD score. You can visualize them in PyMOL.

## Covid Neutralization optimization (Section 4.3)
For this experiment, please also install SRU from here: https://github.com/asappresearch/sru/tree/3.0.0-dev. SRU is used in the covid neutralization predictor. After installation, please run
```
python covid_optimize.py
```
