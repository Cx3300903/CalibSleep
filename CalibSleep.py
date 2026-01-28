# calibsleep_aligned.py
# Final aligned implementation for CalibSleep (paper-consistent)
# - Time-domain encoder outputs sequence features (for CMC time pooling)
# - Time-frequency encoder outputs sequence features (TimesNet-like)
# - CMC: bidirectional cross-attn + time pooling + calibration gate
# - Calibration loss: L_cross (MSE) + L_intra (same-stage pairwise)  [Eq.(9)(10)]
# - Rule-aware transition loss: outer-product of probs with mask M  [Eq.(12)]
# - Total loss: CE + lambda_calib * L_calib + lambda_trans * L_trans [Eq.(13)]

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# Helper: same-label intra-modal cohesion loss (Eq.(10))
# ------------------------------------------------------------
def intra_modal_cohesion_loss(
    feats: torch.Tensor,  # (B, D)
    labels: torch.Tensor,  # (B,)
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Eq.(10)-style: for each class c, compute average pairwise squared distances
    among samples with label c, and sum/average across classes.
    """
    B, D = feats.shape
    unique = labels.unique()
    losses = []
    for c in unique:
        idx = (labels == c).nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() < 2:
            continue
        x = feats[idx]  # (n, D)
        # pairwise squared distances: ||x_i - x_j||^2
        # Efficient: (x^2).sum - 2x x^T + (x^2).sum^T
        x2 = (x * x).sum(dim=1, keepdim=True)  # (n,1)
        dist2 = x2 - 2.0 * (x @ x.t()) + x2.t()  # (n,n)
        # exclude diagonal
        n = x.size(0)
        mask = ~torch.eye(n, device=x.device, dtype=torch.bool)
        losses.append(dist2[mask].mean())
    if len(losses) == 0:
        return feats.new_tensor(0.0)
    return torch.stack(losses).mean()


# ------------------------------------------------------------
# Helper: transition regularization (Eq.(12))
# ------------------------------------------------------------
def transition_regularization_loss(
    probs: torch.Tensor,  # (B, T, K)
    M: torch.Tensor,      # (K, K), 1 for plausible, 0 for implausible (or [0,1] weights)
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Eq.(12): T_t = p_{t-1} p_t^T, penalize mass on (1 - M_ij)
    L_trans = sum_{t} sum_{i,j} T_t(i,j) (1 - M_ij)
    """
    B, T, K = probs.shape
    assert M.shape == (K, K)

    if T < 2:
        return probs.new_tensor(0.0)

    penalty = (1.0 - M).clamp(min=0.0, max=1.0)  # (K,K)

    # For each t: outer product p_{t-1} (B,K) and p_t (B,K) -> (B,K,K)
    p_prev = probs[:, :-1, :]  # (B, T-1, K)
    p_curr = probs[:, 1:, :]   # (B, T-1, K)

    # Compute outer products: (B, T-1, K, 1) * (B, T-1, 1, K) -> (B, T-1, K, K)
    trans = p_prev.unsqueeze(-1) * p_curr.unsqueeze(-2)

    # Weight by penalty mask and sum i,j then average over batch and time
    loss = (trans * penalty).sum(dim=(-1, -2)).mean()  # mean over (B, T-1)
    return loss


# ------------------------------------------------------------
# 2.2 Time-domain encoder (sequence output)
# ------------------------------------------------------------
class TimeDomainEncoderSeq(nn.Module):
    """
    Time-domain encoder aligned with paper:
    - CNN extracts local patterns
    - SENet-style channel attention reweights EEG/EOG contribution
    - BiGRU models temporal dependency
    Output:
      F_s: (B, Ls, D) sequence features (for cross-attention + time pooling)
    Note: To avoid extremely long Ls=3000, we downsample in time by stride.
    """
    def __init__(
        self,
        channels: int = 2,
        cnn_channels=(64, 128, 256),
        stride: int = 10,          # downsample factor on time axis (paper can mention "temporal downsampling")
        gru_hidden: int = 128,
        gru_layers: int = 2,
        d_model: int = 128,
    ):
        super().__init__()
        self.channels = channels
        self.stride = stride
        self.cnn_out = cnn_channels[-1]
        self.d_model = d_model

        layers = []
        in_ch = channels
        for out_ch in cnn_channels:
            k = 3 if out_ch == 64 else 5 if out_ch == 128 else 7
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding="same"),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(out_ch),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        # SENet attention -> 2 weights for EEG/EOG
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.cnn_out, channels),
            nn.Sigmoid()
        )

        # BiGRU over downsampled sequence
        self.bigru = nn.GRU(
            input_size=self.cnn_out,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            bidirectional=True,
            batch_first=True
        )
        self.proj = nn.Linear(gru_hidden * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 2, 3000)
        return F_s: (B, Ls, D)
        """
        B, C, L = x.shape
        assert C == self.channels

        h = self.cnn(x)  # (B, 256, L)

        # channel attention reweight (paper: emphasize EEG/EOG contribution)
        w = self.channel_attn(h)  # (B,2)
        w = w.unsqueeze(-1).unsqueeze(-1)  # (B,2,1,1)

        # Split 256 -> (2,128) then weight and merge back (consistent with your code idea)
        h_split = h.reshape(B, self.channels, -1, L)  # (B,2,128,L)
        h = (h_split * w).reshape(B, self.cnn_out, L)

        # temporal downsampling
        h = h[:, :, :: self.stride]  # (B,256,Ls)
        Ls = h.size(-1)

        # GRU expects (B, Ls, 256)
        h_seq = h.permute(0, 2, 1)
        out, _ = self.bigru(h_seq)        # (B, Ls, 2*hidden)
        F_s = self.proj(out)              # (B, Ls, D)
        return F_s


# ------------------------------------------------------------
# 2.3 Time-frequency encoder (TimesNet-like, sequence output)
# ------------------------------------------------------------
class InceptionConv1D(nn.Module):
    """Multi-kernel temporal conv for periodic patterns (TimesNet-style)."""
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.b1 = nn.Conv1d(d_model, d_hidden, kernel_size=1, padding="same")
        self.b3 = nn.Conv1d(d_model, d_hidden, kernel_size=3, padding="same")
        self.b5 = nn.Conv1d(d_model, d_hidden, kernel_size=5, padding="same")
        self.proj = nn.Conv1d(d_hidden * 3, d_model, kernel_size=1, padding="same")
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> (B, D, T)
        x = x.transpose(1, 2)
        y1 = self.act(self.b1(x))
        y3 = self.act(self.b3(x))
        y5 = self.act(self.b5(x))
        y = torch.cat([y1, y3, y5], dim=1)
        y = self.proj(y)
        return y.transpose(1, 2)


class TimesBlock(nn.Module):
    """Practical TimesNet-like block (keeps sequence length)."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.conv = InceptionConv1D(d_model, d_hidden=max(8, d_model // 2))
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def _fft_enhance(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D)
        Xf = torch.fft.rfft(x, dim=1)
        mag = torch.abs(Xf)
        w = (mag / (mag.mean(dim=1, keepdim=True) + 1e-6)).clamp(0.5, 2.0)
        Xf = Xf * w
        x_rec = torch.fft.irfft(Xf, n=x.size(1), dim=1)
        return x_rec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B,T,D)
        y = self._fft_enhance(self.norm1(x))
        y = self.conv(y)
        x = x + self.drop(y)
        y2 = self.ffn(self.norm2(x))
        x = x + self.drop(y2)
        return x


class TemporalFrequencyEncoderSeq(nn.Module):
    """
    Input x_tf: (B, Lf, F_in)  # e.g., spectrogram columns as tokens (time axis tokens)
    Output F_f: (B, Lf, D)
    """
    def __init__(
        self,
        f_in: int,
        d_model: int = 128,
        n_blocks: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.proj_in = nn.Linear(f_in, d_model)
        self.blocks = nn.ModuleList([TimesBlock(d_model, d_ff, dropout) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_tf: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x_tf)  # (B,Lf,D)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)     # (B,Lf,D)


# ------------------------------------------------------------
# 2.3 CMC module (Eq.(5)(6)(7)(8)) + calibration loss (Eq.(9)(10))
# ------------------------------------------------------------
class CMC(nn.Module):
    """
    Cross-Modal Calibration (CMC):
      1) Bidirectional cross-attention alignment (Eq.(5)(6))
      2) Time pooling -> global vectors (Eq.(7))
      3) Calibration gate -> fusion (Eq.(8))
    """
    def __init__(self, d_model: int = 128, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn_s2f = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_f2s = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_s = nn.LayerNorm(d_model)
        self.norm_f = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        # gate alpha (Eq.(8))
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        self.fuse_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        F_s: torch.Tensor,   # (B,Ls,D)
        F_f: torch.Tensor,   # (B,Lf,D)
        labels: Optional[torch.Tensor] = None,
        lambda_intra: float = 1.0,
        compute_loss: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
          f_fusion: (B, D) fused global feature for classifier
          loss_dict: contains L_cross, L_intra, L_calib, alpha_mean (if compute_loss)
        """
        # Eq.(5): signal queries, tf keys/values
        Qs = self.norm_s(F_s)
        Kf = self.norm_f(F_f)
        Vf = Kf
        Fs_prime, _ = self.attn_s2f(Qs, Kf, Vf, need_weights=False)
        Fs_prime = F_s + self.drop(Fs_prime)  # residual

        # Eq.(6): tf queries, signal keys/values
        Qf = self.norm_f(F_f)
        Ks = self.norm_s(F_s)
        Vs = Ks
        Ff_prime, _ = self.attn_f2s(Qf, Ks, Vs, need_weights=False)
        Ff_prime = F_f + self.drop(Ff_prime)

        # Eq.(7): time pooling to global vectors
        f_s = Fs_prime.mean(dim=1)  # (B,D)
        f_f = Ff_prime.mean(dim=1)  # (B,D)

        # Eq.(8): calibration gate + fusion
        alpha = self.gate(torch.cat([f_s, f_f], dim=-1))  # (B,1)
        f_fusion = self.fuse_norm(alpha * f_s + (1.0 - alpha) * f_f)  # (B,D)

        loss_dict: Dict[str, torch.Tensor] = {"alpha_mean": alpha.mean().detach()}

        if compute_loss and (labels is not None):
            # Eq.(9): cross-modal consistency (same epoch)
            L_cross = F.mse_loss(f_s, f_f)

            # Eq.(10): intra-modal cohesion (same stage within each modality)
            L_intra_s = intra_modal_cohesion_loss(f_s, labels)
            L_intra_f = intra_modal_cohesion_loss(f_f, labels)
            L_intra = 0.5 * (L_intra_s + L_intra_f)

            # Eq.(11): L_calib = L_cross + lambda * L_intra
            L_calib = L_cross + lambda_intra * L_intra

            loss_dict.update({
                "L_cross": L_cross,
                "L_intra": L_intra,
                "L_calib": L_calib
            })

        return f_fusion, loss_dict


# ------------------------------------------------------------
# 2.4 Rule-aware classification head + total loss (Eq.(13))
# ------------------------------------------------------------
class Classifier(nn.Module):
    def __init__(self, d_model: int = 128, hidden: int = 512, n_classes: int = 5, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------------------------------------------------
# CalibSleep (full)
# Supports sequence of epochs to compute transition loss properly
# ------------------------------------------------------------
@dataclass
class CalibSleepConfig:
    n_classes: int = 5
    d_model: int = 128

    # time encoder
    signal_channels: int = 2
    signal_len: int = 3000
    cnn_channels: Tuple[int, int, int] = (64, 128, 256)
    stride: int = 10
    gru_hidden: int = 128
    gru_layers: int = 2

    # tf encoder
    tf_feat_in: int = 128    # you define based on how you tokenize spectrogram columns
    tf_blocks: int = 2
    tf_dff: int = 256

    # cmc
    n_heads: int = 4
    dropout: float = 0.1

    # loss weights (Eq.(13) + Eq.(11))
    lambda_calib: float = 1.0
    lambda_intra: float = 1.0   # the lambda inside L_calib
    lambda_trans: float = 0.1


class CalibSleep(nn.Module):
    """
    Inputs:
      x_sig: (B, T, 2, 3000)  # a sequence of T epochs for transition regularization
      x_tf:  (B, T, Lf, F_in) # time-frequency tokens per epoch
      y:     (B, T)           # labels per epoch
    Outputs:
      logits: (B, T, K)
      probs:  (B, T, K)
      losses dict (if y provided)
    """
    def __init__(self, cfg: CalibSleepConfig, M: torch.Tensor):
        super().__init__()
        self.cfg = cfg

        self.time_enc = TimeDomainEncoderSeq(
            channels=cfg.signal_channels,
            cnn_channels=cfg.cnn_channels,
            stride=cfg.stride,
            gru_hidden=cfg.gru_hidden,
            gru_layers=cfg.gru_layers,
            d_model=cfg.d_model
        )
        self.tf_enc = TemporalFrequencyEncoderSeq(
            f_in=cfg.tf_feat_in,
            d_model=cfg.d_model,
            n_blocks=cfg.tf_blocks,
            d_ff=cfg.tf_dff,
            dropout=cfg.dropout
        )
        self.cmc = CMC(d_model=cfg.d_model, n_heads=cfg.n_heads, dropout=cfg.dropout)
        self.clf = Classifier(d_model=cfg.d_model, hidden=512, n_classes=cfg.n_classes, dropout=0.5)

        self.register_buffer("M", M.float())  # (K,K)

    def forward(
        self,
        x_sig: torch.Tensor,   # (B,T,2,3000)
        x_tf: torch.Tensor,    # (B,T,Lf,F_in)
        y: Optional[torch.Tensor] = None,  # (B,T)
        compute_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        B, T, C, L = x_sig.shape
        assert C == self.cfg.signal_channels and L == self.cfg.signal_len

        # Flatten epochs to run encoders
        x_sig_flat = x_sig.reshape(B * T, C, L)                # (B*T,2,3000)
        x_tf_flat = x_tf.reshape(B * T, x_tf.size(2), x_tf.size(3))  # (B*T,Lf,F_in)

        # Encode sequences
        F_s = self.time_enc(x_sig_flat)  # (B*T,Ls,D)
        F_f = self.tf_enc(x_tf_flat)     # (B*T,Lf,D)

        # CMC fusion (needs labels per epoch)
        if y is not None:
            y_flat = y.reshape(B * T)
        else:
            y_flat = None

        f_fusion, cmc_loss = self.cmc(
            F_s, F_f,
            labels=y_flat,
            lambda_intra=self.cfg.lambda_intra,
            compute_loss=(compute_loss and (y_flat is not None))
        )  # (B*T,D)

        logits = self.clf(f_fusion)            # (B*T,K)
        probs = F.softmax(logits, dim=-1)      # (B*T,K)

        logits_seq = logits.reshape(B, T, self.cfg.n_classes)
        probs_seq = probs.reshape(B, T, self.cfg.n_classes)

        out: Dict[str, torch.Tensor] = {
            "logits": logits_seq,
            "probs": probs_seq,
            "alpha_mean": cmc_loss.get("alpha_mean", torch.tensor(0.0, device=logits.device)),
        }

        # Losses
        if compute_loss and (y is not None):
            # CE over all epochs
            L_ce = F.cross_entropy(logits_seq.reshape(B * T, -1), y.reshape(B * T))

            # L_calib from CMC (Eq.(11))
            L_calib = cmc_loss.get("L_calib", logits.new_tensor(0.0))

            # Transition loss (Eq.(12))
            L_trans = transition_regularization_loss(probs_seq, self.M)

            # Total (Eq.(13))
            L_total = L_ce + self.cfg.lambda_calib * L_calib + self.cfg.lambda_trans * L_trans

            out.update({
                "L_total": L_total,
                "L_ce": L_ce.detach(),
                "L_cross": cmc_loss.get("L_cross", logits.new_tensor(0.0)).detach(),
                "L_intra": cmc_loss.get("L_intra", logits.new_tensor(0.0)).detach(),
                "L_calib": L_calib.detach(),
                "L_trans": L_trans.detach(),
            })

        return out
