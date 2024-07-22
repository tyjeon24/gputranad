"""TranAD and PyTorch Lightning wrapper."""

import math

import lightning as L
import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerEncoder
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

N_WINDOWS = 10
torch.manual_seed(1)


class PositionalEncoding(nn.Module):
    """Positional Encoding layer."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """Init model.

        Args:
            d_model (int): Dimension of model, usually d_model = 2 * (the number of data column).
            dropout (float, optional): Dropout ratio. Defaults to 0.1.
            max_len (int, optional): Window length. Defaults to 5000.

        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.tensor, pos: int = 0):
        """Forward pass.

        Args:
            x (torch.tensor): Input.
            pos (int, optional): Starting position. Defaults to 0.

        Returns:
            torch.tensor: X with the positional encodings.

        """
        x = x + self.pe[pos : pos + x.size(0), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """Encoder layer with Attention + Feed forward."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward=16, dropout=0):
        """Initialize .

        Args:
            d_model (int): Dimension of model.
            nhead (int): The number of heads.
            dim_feedforward (int, optional): The hidden layer of feed forward network. Defaults to 16.
            dropout (int, optional): Dropout ratio. Defaults to 0.

        """
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, src: torch.tensor, src_mask=None, src_key_padding_mask=None, is_causal=None):
        """Forward pass.

        Args:
            src (torch.tensor): Input.
            src_mask (torch.tensor, optional): Mask for tensor. Defaults to None.
            src_key_padding_mask (torch.tensor, optional): Added because of pyTorch version problem. Defaults to None.
            is_causal (bool, optional): Added because of pyTorch version problem. Defaults to None.

        Returns:
            torch.tensor: Tensor.

        """
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoderLayer(nn.Module):
    """Decoder layer with Attention + Feed forward."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 16, dropout: float = 0.0):
        """Initialize.

        Args:
            d_model (int): Dimension of model.
            nhead (int): The number of heads.
            dim_feedforward (int, optional): The hidden layer of feed forward network. Defaults to 16.
            dropout (float, optional): Dropout ratio. Defaults to 0.0.

        """
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        tgt_is_causal=None,
        memory_is_causal=None,
    ):
        """Forward pass.

        Args:
            tgt (torch.tensor): Target tensor.
            memory (torch.tensor): Tensor from encoding layer.
            tgt_mask (torch.tensor, optional): Mask for tensor.
            memory_mask (torch.tensor, optional): Mask for tensor.
            tgt_key_padding_mask (torch.tensor, optional): Added because of pyTorch version problem.
            memory_key_padding_mask (torch.tensor, optional): Added because of pyTorch version problem.
            tgt_is_causal (bool, optional): Added because of pyTorch version problem.
            memory_is_causal (bool, optional): Added because of pyTorch version problem.

        Returns:
            torch.tensor: Tensor.

        """
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class TranAD(nn.Module):
    """TranAD model."""

    def __init__(self, n_feats: int):
        """Initialize.

        Args:
            n_feats (int): Number of data columns.

        """
        super(TranAD, self).__init__()
        self.name = "TranAD"
        self.n_feats = n_feats
        self.n_window = N_WINDOWS
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(2 * n_feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=2 * n_feats, nhead=n_feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers1 = TransformerDecoderLayer(d_model=2 * n_feats, nhead=n_feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(d_model=2 * n_feats, nhead=n_feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * n_feats, n_feats), nn.Sigmoid())

    def encode(self, src: torch.tensor, c: torch.tensor, tgt: torch.tensor):
        """Encode tensor.

        Args:
            src (torch.tensor): Source.
            c (torch.tensor): Context.
            tgt (torch.tensor): Target.

        Returns:
            tuple: Encoded tensor tuple.

        """
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src: torch.tensor, tgt: torch.tensor):
        """Forward pass.

        Args:
            src (torch.tensor): Source.
            tgt (torch.tensor): Target.

        Returns:
            torch.tensor: Prediction values.

        """
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2


class TranADLitModel(L.LightningModule):
    """LightningModule for training TranAD."""

    def __init__(self, model: nn.Module):
        """Initialize LightningModule.

        model (nn.Module): TranAD model.

        """
        super(TranADLitModel, self).__init__()
        self.model = model
        self.loss_fn = nn.MSELoss(reduction="mean")

    def forward(self, src: torch.tensor, tgt: torch.tensor):
        """Forward pass.

        Args:
            src (torch.tensor): Source.
            tgt (torch.tensor): Target.

        Returns:
            tuple: TranAD prediction.

        """
        return self.model(src, tgt)

    def training_step(self, batch: torch.tensor):
        """Train using dataloader or batch data.

        Args:
            batch (torch.tensor): X input

        Returns:
            torch.tensor: loss for backpropagation.

        """
        d, _ = batch
        epoch = self.current_epoch + 1
        local_bs = d.shape[0]
        window = d.permute(1, 0, 2)
        elem = window[-1, :, :].view(1, local_bs, self.model.n_feats)

        z = self(window, elem)

        if isinstance(z, tuple):
            loss = (1 / epoch) * self.loss_fn(z[0], elem) + (1 - 1 / epoch) * self.loss_fn(z[1], elem)
            z = z[1]
        else:
            loss = self.loss_fn(z, elem)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        """Set Optimizers.

        Returns:
            List: optimizers with scheduler.

        """
        optimizer = AdamW(self.parameters(), lr=0.0001, weight_decay=1e-5)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.9)
        return [optimizer], [scheduler]
