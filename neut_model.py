from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import Tensor
from pytorch_lightning.callbacks import ModelCheckpoint

AA_VOCAB = {
    "#": 0,
    "A": 1,
    "R": 2,
    "N": 3,
    "D": 4,
    "C": 5,
    "Q": 6,
    "E": 7,
    "G": 8,
    "H": 9,
    "I": 10,
    "L": 11,
    "K": 12,
    "M": 13,
    "F": 14,
    "P": 15,
    "S": 16,
    "T": 17,
    "W": 18,
    "Y": 19,
    "V": 20,
    "X": 21,
    "-": 22,
    "O": 23,
    "*": 24,
}
RELEVANT_VIRUSES = {"SARS-CoV1", "SARS-CoV2"}
RELEVANT_KEYS = {
    "Neutralising Vs",
    "Not Neutralising Vs",
    "Binds to",
    "Doesn't Bind to",
}
TYPE_MAP = {
    "S1; non-RBD": "ntd",
    "S2 (quaternary glycan epitope)": "s2",
    "S: NTD": "ntd",
    "S: RBD": "rbd",
    "S; NTD": "ntd",
    "S; Possibly RBD": "rbd",
    "S; RBD": "rbd",
    "S; RBD/non-RBD": "unk",
    "S; S1": "unk",
    "S; S1 non-RBD": "ntd",
    "S; S1/S2": "unk",
    "S; S1/S2 Cleavage Site": "unk",
    "S; S2": "s2",
    "S; S2 (quaternary glycan epitope)": "s2",
    "S; S2 Stem Helix": "s2",
    "S; Unk": "unk",
    "S; non-RBD": "unk",
    "S; non-S1": "s2",
    "S; probably RBD (implied by clustering)": "rbd",
}


class RNNEncoder(nn.Module):
    """Implements a multi-layer RNN.

    This module can be used to create multi-layer RNN models, and
    provides a way to reduce to output of the RNN to a single hidden
    state by pooling the encoder states either by taking the maximum,
    average, or by taking the last hidden state before padding.
    Padding is dealt with by using torch's PackedSequence.

    Attributes
    ----------
    rnn: nn.Module
        The rnn submodule

    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_layers: int = 1,
        rnn_type: str = "lstm",
        dropout: float = 0,
        attn_dropout: float = 0,
        attn_heads: int = 1,
        bidirectional: bool = False,
        layer_norm: bool = False,
        highway_bias: float = -2,
        rescale: bool = True,
        enforce_sorted: bool = False,
        **kwargs,
    ) -> None:
        """Initializes the RNNEncoder object.
        Parameters
        ----------
        input_size : int
            The dimension the input data
        hidden_size : int
            The hidden dimension to encode the data in
        n_layers : int, optional
            The number of rnn layers, defaults to 1
        rnn_type : str, optional
           The type of rnn cell, one of: `lstm`, `gru`, `sru`
           defaults to `lstm`
        dropout : float, optional
            Amount of dropout to use between RNN layers, defaults to 0
        bidirectional : bool, optional
            Set to use a bidrectional encoder, defaults to False
        layer_norm : bool, optional
            [SRU only] whether to use layer norm
        highway_bias : float, optional
            [SRU only] value to use for the highway bias
        rescale : bool, optional
            [SRU only] whether to use rescaling
        enforce_sorted: bool
            Whether rnn should enforce that sequences are ordered by
            length. Requires True for ONNX support. Defaults to False.
        kwargs
            Additional parameters to be passed to SRU when building
            the rnn.
        Raises
        ------
        ValueError
            The rnn type should be one of: `lstm`, `gru`, `sru`

        """
        super().__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.enforce_sorted = enforce_sorted
        self.output_size = 2 * hidden_size if bidirectional else hidden_size

        if rnn_type in ["lstm", "gru"]:
            rnn_fn = nn.LSTM if rnn_type == "lstm" else nn.GRU
            self.rnn = rnn_fn(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=n_layers,
                dropout=dropout,
                bidirectional=bidirectional,
            )
        elif rnn_type == "sru":
            from sru import SRU

            try:
                self.rnn = SRU(
                    input_size,
                    hidden_size,
                    num_layers=n_layers,
                    dropout=dropout,
                    bidirectional=bidirectional,
                    layer_norm=layer_norm,
                    rescale=rescale,
                    highway_bias=highway_bias,
                    **kwargs,
                )
            except TypeError:
                raise ValueError(f"Unkown kwargs passed to SRU: {kwargs}")
        elif rnn_type == "srupp":
            from sru import SRUpp

            try:
                self.rnn = SRUpp(
                    input_size,
                    hidden_size,
                    hidden_size // 2,
                    num_layers=n_layers,
                    highway_bias=highway_bias,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    num_heads=attn_heads,
                    layer_norm=layer_norm,
                    attn_layer_norm=True,
                    bidirectional=bidirectional,
                    **kwargs,
                )
            except TypeError:
                raise ValueError(f"Unkown kwargs passed to SRU: {kwargs}")
        else:
            raise ValueError(f"Unkown rnn type: {rnn_type}, use of of: gru, sru, lstm")

    def forward(
        self,
        data: Tensor,
        state: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Performs a forward pass through the network.
        Parameters
        ----------
        data : Tensor
            The input data, as a float tensor of shape [B x S x E]
        state: Tensor
            An optional previous state of shape [L x B x H]
        padding_mask: Tensor, optional
            The padding mask of shape [B x S], dtype should be bool
        Returns
        -------
        Tensor
            The encoded output, as a float tensor of shape [B x S x H]
        Tensor
            The encoded state, as a float tensor of shape [L x B x H]

        """
        data = data.transpose(0, 1)
        if padding_mask is not None:
            padding_mask = padding_mask.transpose(0, 1)

        if padding_mask is None:
            # Default RNN behavior
            output, state = self.rnn(data, state)
        elif self.rnn_type == "sru":
            # SRU takes a mask instead of PackedSequence objects
            # ~ operator negates bool tensor in torch 1.3
            output, state = self.rnn(data, state, mask_pad=(~padding_mask))
        elif self.rnn_type == "srupp":
            # SRU takes a mask instead of PackedSequence objects
            # ~ operator negates bool tensor in torch 1.3
            output, state, _ = self.rnn(data, state, mask_pad=(~padding_mask))
        else:
            # Deal with variable length sequences
            lengths = padding_mask.long().sum(dim=0)
            # Pass through the RNN
            packed = nn.utils.rnn.pack_padded_sequence(
                data, lengths.cpu(), enforce_sorted=self.enforce_sorted
            )
            output, state = self.rnn(packed, state)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, total_length=data.size(0))

        # TODO investigate why PyTorch returns type Any for output
        return output.transpose(0, 1).contiguous(), state  # type: ignore


class SRUppModel(nn.Module):

    def __init__(
        self,
        num_aa: int,
        num_tokens: int,
        n_layers: int = 1,
        hidden_dim: int = 256,
        dropout: float = 0,
        ab_pad_id: int = 0,
        virus_pad_id: int = 0,
        use_srupp: bool = False,
    ):
        super().__init__()

        # Virus encoder
        self.hidden_dim = hidden_dim
        self.seq_embedding = nn.Embedding(num_aa, hidden_dim // 4)

        # Antibody encoder
        rnn_type = "srupp" if use_srupp else "sru"
        self.dropout = nn.Dropout(dropout)
        self.rnn_ab = RNNEncoder(
            hidden_dim // 4,
            hidden_dim // 2,
            n_layers=n_layers,
            rnn_type=rnn_type,
            bidirectional=True,
            dropout=dropout,
        )
        self.ab_pad_id = ab_pad_id
        self.virus_pad_id = virus_pad_id
        self.num_tokens = num_tokens

    @property
    def output_dim(self):
        return self.hidden_dim

    def forward(self, ab_seq):
        # Compute padding mask
        padding_ab = ab_seq != self.ab_pad_id

        # Compute token embeddings
        ab_emb = self.dropout(self.seq_embedding(ab_seq))

        # Pass through SRUpp layers
        ab_encodings, _ = self.rnn_ab(ab_emb, padding_mask=padding_ab)
        return ab_encodings


class MultiABOnlyCoronavirusModel(pl.LightningModule):

    def __init__(
        self,
        num_aa: int,
        num_tokens: int,
        n_layers: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0,
        lr: float = 1e-3,
        ab_pad_id: int = 0,
        virus_pad_id: int = 0,
        neut_lambda: float = 0.5,
        use_srupp: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.ab_pad_id = ab_pad_id
        self.virus_pad_id = virus_pad_id
        self.neut_lambda = neut_lambda
        self.dropout = nn.Dropout(dropout)
        self.encoder = SRUppModel(  # type: ignore
            num_aa=num_aa,
            num_tokens=num_tokens,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            dropout=dropout,
            ab_pad_id=ab_pad_id,
            virus_pad_id=virus_pad_id,
            use_srupp=use_srupp,
        )
        encoder_dim = self.encoder.output_dim
        self.fc_neut = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dim // 2, 2),
        )
        #self.neut_auc_sars2 = AUROCWithMask(
        #    num_classes=2, average=None, compute_on_step=False
        #)
        #self.neut_auc_sars1 = AUROCWithMask(
        #    num_classes=2, average=None, compute_on_step=False
        #)

    @classmethod
    def add_extra_args(cls) -> Dict:
        extra_args = {
            "num_aa": len(AA_VOCAB),
            "num_tokens": 1024,
            "ab_pad_id": AA_VOCAB["#"],
            "virus_pad_id": AA_VOCAB["#"],
        }
        return extra_args

    def average(self, data, padding):
        data = (data * padding.unsqueeze(2)).sum(dim=1)
        padding_sum = padding.sum(dim=1)
        padding_sum[padding_sum == 0] = 1.0
        avg = data / padding_sum.unsqueeze(1)
        return avg

    def forward(self, ab_seq):
        padding_mask_ab = (ab_seq != self.ab_pad_id).float()
        ab_encodings = self.encoder(ab_seq)
        output_encoding = self.average(ab_encodings, padding_mask_ab)
        output_encoding = self.dropout(output_encoding)
        neut_logits = self.fc_neut(output_encoding).squeeze(1)

        return neut_logits

    def configure_callbacks(self):
        return [ModelCheckpoint(monitor="auc", save_top_k=1, mode="max")]

    def compute_metrics(self, batch):
        ab_seq = batch["ab"]
        neut_label = batch["neut_label"]
        neut_mask = batch["neut_mask"]
        neut_logits = self(ab_seq)
        neut_loss = F.binary_cross_entropy_with_logits(
            neut_logits, neut_label.float(), reduction="none"
        )
        neut_mask_sum = neut_mask.sum()
        neut_mask_sum = neut_mask_sum if neut_mask_sum > 0 else 1.0
        neut_loss = (neut_loss * neut_mask).sum() / neut_mask_sum

        # Final loss
        loss = neut_loss

        # Compute metrics (ignore neg label 0)
        self.neut_auc_sars1(
            torch.sigmoid(neut_logits[:, 0]),
            neut_label[:, 0].long(),
            neut_mask[:, 0].bool(),
        )
        self.neut_auc_sars2(
            torch.sigmoid(neut_logits[:, 1]),
            neut_label[:, 1].long(),
            neut_mask[:, 1].bool(),
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_metrics(batch)
        self.log("loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_metrics(batch)
        self.log("val_loss", loss, prog_bar=True)

    def training_epoch_end(self, output):
        try:
            neut_auc_sars1 = self.neut_auc_sars1.compute()
        except Exception:
            neut_auc_sars1 = 0.5

        try:
            neut_auc_sars2 = self.neut_auc_sars2.compute()
        except Exception:
            neut_auc_sars2 = 0.5

        self.log("train_auc_sars_cov_1", neut_auc_sars1, prog_bar=False)
        self.log("train_auc_sars_cov_2", neut_auc_sars2, prog_bar=False)
        self.neut_auc_sars1.reset()
        self.neut_auc_sars2.reset()

    def validation_epoch_end(self, output):
        try:
            neut_auc_sars1 = self.neut_auc_sars1.compute()
        except Exception as e:
            print(e)
            neut_auc_sars1 = 0.5

        try:
            neut_auc_sars2 = self.neut_auc_sars2.compute()
        except Exception:
            neut_auc_sars2 = 0.5

        self.log("auc", (neut_auc_sars1 + neut_auc_sars2) / 2, prog_bar=True)
        self.log("auc_sars_cov_1", neut_auc_sars1, prog_bar=True)
        self.log("auc_sars_cov_2", neut_auc_sars2, prog_bar=True)
        self.neut_auc_sars1.reset()
        self.neut_auc_sars2.reset()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, output):
        return self.validation_epoch_end(output)

    def configure_optimizers(self):
        return RAdam((p for p in self.parameters() if p.requires_grad), lr=self.lr)
