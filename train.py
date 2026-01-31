import os
import math
import json
import time
import random
import argparse
from dataclasses import asdict
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------
# Import model (your uploaded CalibSleep.py)
# ------------------------------------------------------------
from CalibSleep import CalibSleep, CalibSleepConfig


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def compute_metrics_from_confmat(cm: np.ndarray) -> Dict[str, float]:
    """
    cm: (K,K) confusion matrix counts
    Returns: Acc, Macro-F1, Kappa
    """
    K = cm.shape[0]
    total = cm.sum()
    acc = np.trace(cm) / (total + 1e-12)

    # per-class precision/recall/f1
    f1s = []
    for k in range(K):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        f1s.append(f1)
    mf1 = float(np.mean(f1s))

    # Cohen's kappa
    row_m = cm.sum(axis=1)
    col_m = cm.sum(axis=0)
    pe = (row_m @ col_m) / ((total ** 2) + 1e-12)
    kappa = float((acc - pe) / (1 - pe + 1e-12))
    return {"acc": float(acc), "mf1": mf1, "kappa": kappa}


@torch.no_grad()
def confusion_matrix_counts(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


# ------------------------------------------------------------
# Transition mask M (Rule-aware)
# ------------------------------------------------------------
def build_transition_mask_aasm_like(n_classes: int = 5, soften: float = 1.0) -> torch.Tensor:
    """
    Build a plausible-transition mask M in [0,1], shape (K,K).
    K=5 stages: [W, N1, N2, N3, REM]
    - Allow self-transition
    - Allow adjacent / clinically plausible transitions
    You can adjust it for your exact rule design.

    soften=1.0 => hard mask {0,1}
    soften<1.0 => allow weak penalty (e.g., set implausible to soften instead of 0)
    """
    assert n_classes == 5, "This default mask assumes 5-stage AASM: W,N1,N2,N3,REM"
    W, N1, N2, N3, REM = 0, 1, 2, 3, 4

    M = np.zeros((5, 5), dtype=np.float32)

    # self transitions
    np.fill_diagonal(M, 1.0)

    # plausible transitions (common scoring / physiology-inspired)
    # W <-> N1
    M[W, N1] = 1.0
    M[N1, W] = 1.0

    # N1 <-> N2
    M[N1, N2] = 1.0
    M[N2, N1] = 1.0

    # N2 <-> N3
    M[N2, N3] = 1.0
    M[N3, N2] = 1.0

    # N2 <-> REM (common)
    M[N2, REM] = 1.0
    M[REM, N2] = 1.0

    # N1 <-> REM (can happen, but less common; keep as plausible if you want)
    M[N1, REM] = 1.0
    M[REM, N1] = 1.0

    # Optional: W <-> REM is rare; default set to 0 (penalize)
    # Optional: N3 <-> REM is implausible; keep 0 (penalize)

    if soften < 1.0:
        # Instead of 0, set implausible entries to soften (e.g., 0.1)
        M = np.where(M > 0, M, soften).astype(np.float32)

    return torch.from_numpy(M)


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
class SleepSequenceDataset(Dataset):
    """
    Expects one file per subject OR one merged file.
    Supported formats:
      - .npz: contains arrays:
          x_sig: (N,2,3000)
          x_tf:  (N,Lf,F_in)
          y:     (N,)
      - .pt:  torch dict with same keys

    This dataset groups consecutive epochs into sequences of length T:
      returns x_sig_seq: (T,2,3000), x_tf_seq: (T,Lf,F_in), y_seq: (T,)

    If subject boundaries exist, you should store each subject as one file to avoid crossing subjects.
    """
    def __init__(
        self,
        data_dir: str,
        seq_len: int = 5,
        stride: int = 1,
        file_ext: str = ".npz",
        max_subjects: Optional[int] = None,
        subject_ids: Optional[List[str]] = None,
    ):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.stride = stride
        self.file_ext = file_ext

        files = sorted([f for f in os.listdir(data_dir) if f.endswith(file_ext)])
        if subject_ids is not None:
            # filter by given IDs (filename contains id)
            files = [f for f in files if any(sid in f for sid in subject_ids)]
        if max_subjects is not None:
            files = files[:max_subjects]

        if len(files) == 0:
            raise FileNotFoundError(f"No *{file_ext} files found in {data_dir}")

        self.files = [os.path.join(data_dir, f) for f in files]

        # Build an index of (file_idx, start_epoch)
        self.index = []
        self._cache = {}  # optional small cache

        for fi, fp in enumerate(self.files):
            x_sig, x_tf, y = self._load_file(fp)
            N = int(y.shape[0])
            # sequences: start from 0 to N-seq_len
            for s in range(0, N - seq_len + 1, stride):
                self.index.append((fi, s))

    def _load_file(self, fp: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if fp in self._cache:
            return self._cache[fp]

        if fp.endswith(".npz"):
            pack = np.load(fp, allow_pickle=False)
            x_sig = pack["x_sig"].astype(np.float32)
            x_tf = pack["x_tf"].astype(np.float32)
            y = pack["y"].astype(np.int64)
        elif fp.endswith(".pt"):
            d = torch.load(fp, map_location="cpu")
            x_sig = d["x_sig"].numpy().astype(np.float32)
            x_tf = d["x_tf"].numpy().astype(np.float32)
            y = d["y"].numpy().astype(np.int64)
        else:
            raise ValueError(f"Unsupported file format: {fp}")

        self._cache[fp] = (x_sig, x_tf, y)
        return x_sig, x_tf, y

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fi, s = self.index[idx]
        fp = self.files[fi]
        x_sig, x_tf, y = self._load_file(fp)

        xs = x_sig[s:s + self.seq_len]  # (T,2,3000)
        xt = x_tf[s:s + self.seq_len]   # (T,Lf,F_in)
        yy = y[s:s + self.seq_len]      # (T,)

        return {
            "x_sig": torch.from_numpy(xs),  # (T,2,3000)
            "x_tf": torch.from_numpy(xt),   # (T,Lf,F_in)
            "y": torch.from_numpy(yy),      # (T,)
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    x_sig = torch.stack([b["x_sig"] for b in batch], dim=0)  # (B,T,2,3000)
    x_tf = torch.stack([b["x_tf"] for b in batch], dim=0)    # (B,T,Lf,F_in)
    y = torch.stack([b["y"] for b in batch], dim=0)          # (B,T)
    return {"x_sig": x_sig, "x_tf": x_tf, "y": y}


# ------------------------------------------------------------
# Train / Eval
# ------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    grad_clip: float = 1.0
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    n_steps = 0

    meter = {"L_total": 0.0, "L_ce": 0.0, "L_calib": 0.0, "L_trans": 0.0, "alpha_mean": 0.0}

    for batch in loader:
        x_sig = batch["x_sig"].to(device)
        x_tf = batch["x_tf"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                out = model(x_sig=x_sig, x_tf=x_tf, y=y, compute_loss=True)
                loss = out["L_total"]
            scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x_sig=x_sig, x_tf=x_tf, y=y, compute_loss=True)
            loss = out["L_total"]
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = x_sig.size(0)
        total_loss += float(loss.item())
        n_steps += 1

        meter["L_total"] += float(out["L_total"].item())
        meter["L_ce"] += float(out["L_ce"].item()) if "L_ce" in out else 0.0
        meter["L_calib"] += float(out["L_calib"].item()) if "L_calib" in out else 0.0
        meter["L_trans"] += float(out["L_trans"].item()) if "L_trans" in out else 0.0
        meter["alpha_mean"] += float(out["alpha_mean"].item()) if "alpha_mean" in out else 0.0

    for k in meter:
        meter[k] /= max(n_steps, 1)
    return meter


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_classes: int = 5,
    export_preds: bool = False
) -> Dict[str, object]:
    model.eval()
    all_true = []
    all_pred = []
    all_prob = []
    alpha_means = []

    total_loss = 0.0
    n_steps = 0

    for batch in loader:
        x_sig = batch["x_sig"].to(device)
        x_tf = batch["x_tf"].to(device)
        y = batch["y"].to(device)

        out = model(x_sig=x_sig, x_tf=x_tf, y=y, compute_loss=True)
        probs = out["probs"]  # (B,T,K)
        logits = out["logits"]

        # flatten to epochs
        y_true = y.reshape(-1).detach().cpu().numpy()
        y_pred = probs.argmax(dim=-1).reshape(-1).detach().cpu().numpy()

        all_true.append(y_true)
        all_pred.append(y_pred)

        if export_preds:
            all_prob.append(probs.reshape(-1, n_classes).detach().cpu().numpy())

        if "alpha_mean" in out:
            alpha_means.append(float(out["alpha_mean"].item()))

        if "L_total" in out:
            total_loss += float(out["L_total"].item())
            n_steps += 1

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)

    cm = confusion_matrix_counts(y_true, y_pred, n_classes=n_classes)
    metrics = compute_metrics_from_confmat(cm)

    res: Dict[str, object] = {
        "loss": total_loss / max(n_steps, 1),
        "cm": cm,
        "metrics": metrics,
        "alpha_mean": float(np.mean(alpha_means)) if len(alpha_means) else 0.0,
    }

    if export_preds:
        res["y_true"] = y_true
        res["y_pred"] = y_pred
        res["probs"] = np.concatenate(all_prob, axis=0) if len(all_prob) else None

    return res


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("CalibSleep Training (paper-aligned)")

    # data
    parser.add_argument("--train_dir", type=str, required=True, help="Directory with train subject files (.npz or .pt)")
    parser.add_argument("--val_dir", type=str, required=True, help="Directory with val subject files (.npz or .pt)")
    parser.add_argument("--test_dir", type=str, default=None, help="Optional test dir for final evaluation")
    parser.add_argument("--file_ext", type=str, default=".npz", choices=[".npz", ".pt"])
    parser.add_argument("--seq_len", type=int, default=5, help="T (epochs per sequence) for transition regularization")
    parser.add_argument("--seq_stride", type=int, default=1, help="stride when forming sequences")

    # train
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", help="use mixed precision")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)

    # loss weights (paper-aligned defaults)
    parser.add_argument("--lambda_calib", type=float, default=1.0)
    parser.add_argument("--lambda_intra", type=float, default=1.0)
    parser.add_argument("--lambda_trans", type=float, default=0.1)

    # model dims
    parser.add_argument("--tf_feat_in", type=int, required=True, help="F_in of x_tf tokens (must match your preprocessed data)")
    parser.add_argument("--signal_len", type=int, default=3000)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--run_name", type=str, default="calibsleep")

    # rule mask options
    parser.add_argument("--mask_soften", type=float, default=1.0, help="1.0=hard mask, e.g. 0.1=soft mask for implausible")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build transition mask M
    M = build_transition_mask_aasm_like(n_classes=5, soften=args.mask_soften).to(device)

    # Build config (aligned with your CalibSleep.py)
    cfg = CalibSleepConfig(
        n_classes=5,
        d_model=128,
        signal_channels=2,
        signal_len=args.signal_len,
        tf_feat_in=args.tf_feat_in,
        lambda_calib=args.lambda_calib,
        lambda_intra=args.lambda_intra,
        lambda_trans=args.lambda_trans,
    )

    # Model
    model = CalibSleep(cfg, M=M).to(device)

    # Optimizer (paper: AdamW, lr=1e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None

    # Data
    train_ds = SleepSequenceDataset(args.train_dir, seq_len=args.seq_len, stride=args.seq_stride, file_ext=args.file_ext)
    val_ds = SleepSequenceDataset(args.val_dir, seq_len=args.seq_len, stride=args.seq_stride, file_ext=args.file_ext)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        drop_last=False
    )

    # Save config
    with open(os.path.join(args.save_dir, f"{args.run_name}_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.save_dir, f"{args.run_name}_modelcfg.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    best_val = -1.0
    best_path = os.path.join(args.save_dir, f"{args.run_name}_best.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr = train_one_epoch(model, train_loader, optimizer, device, scaler=scaler, grad_clip=args.grad_clip)
        va = evaluate(model, val_loader, device, n_classes=5, export_preds=False)

        dt = time.time() - t0
        msg = (
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"train: L={tr['L_total']:.4f} (CE={tr['L_ce']:.4f}, calib={tr['L_calib']:.4f}, trans={tr['L_trans']:.4f}), "
            f"alpha={tr['alpha_mean']:.4f} | "
            f"val: acc={va['metrics']['acc']:.4f}, mf1={va['metrics']['mf1']:.4f}, kappa={va['metrics']['kappa']:.4f}, "
            f"loss={va['loss']:.4f}, alpha={va['alpha_mean']:.4f} | {dt:.1f}s"
        )
        print(msg)

        # track best on val MF1 (you can switch to acc)
        cur = va["metrics"]["mf1"]
        if cur > best_val:
            best_val = cur
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val_mf1": best_val,
                    "cfg": asdict(cfg),
                    "args": vars(args),
                },
                best_path
            )
            print(f"  -> saved best checkpoint to {best_path} (val_mf1={best_val:.4f})")

    # Final test evaluation (optional)
    if args.test_dir is not None:
        print("\n[Testing best checkpoint]")
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])

        test_ds = SleepSequenceDataset(args.test_dir, seq_len=args.seq_len, stride=args.seq_stride, file_ext=args.file_ext)
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_fn,
            drop_last=False
        )

        te = evaluate(model, test_loader, device, n_classes=5, export_preds=True)
        print(f"Test: acc={te['metrics']['acc']:.4f}, mf1={te['metrics']['mf1']:.4f}, kappa={te['metrics']['kappa']:.4f}")

        # export predictions for confusion matrix / interpretability
        np.savez(
            os.path.join(args.save_dir, f"{args.run_name}_test_preds.npz"),
            y_true=te["y_true"],
            y_pred=te["y_pred"],
            probs=te["probs"],
            cm=te["cm"]
        )
        print(f"Saved test predictions to {os.path.join(args.save_dir, f'{args.run_name}_test_preds.npz')}")

    print("\nDone.")


if __name__ == "__main__":
    main()
