#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
#  train_v7.py – 2025-08-10 (EWS 논문용, v7 철학 반영)                           #
#  • v7 데이터셋과 호환 (time_index, area_order, label_stats 등 활용)            #
#  • 실험 모드:
#     - next_year : (Y0~Y1)로 학습/튜닝 → (Y1+1) 평가 (논문 실험 설계)          #
#     - fixed     : build에서 저장된 splits.pt(train/val/test) 사용             #
#     - walkforward : 오리진별(월) 학습/튜닝 → 해당 월 OOS                       #
#  • 입력 길이 L 가변 (--input-len), 마스크 정책(all/any)                        #
#  • 임계 정책(F1/Youden/Budget/Cost), 캘리브레이션(Platt/Isotonic)              #
#  • 월별 OOS 곡선/예산 기반 지표/예측 저장                                      #
#  • 멀티-임계 라벨 선택 지원(--label-suffix, 예: 'thr2.5')                      #
# ---------------------------------------------------------------------------- #

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import argparse, json, logging, random, time
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm

#from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from datetime import datetime
import platform, sys, subprocess, hashlib, shutil, inspect

from model_v7 import build_model_v7  # 같은 폴더에 두세요

LOG = logging.getLogger("train_v7")
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

# ─────────────────────── 1) CLI ─────────────────────────────────────────── #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ----- 필수 경로 ----- #
    p.add_argument("--proc-dir",   type=Path, required=True,
                   help="ex) data/processed_v7_ifpa/20250810")
    p.add_argument("--save-dir",   type=Path,  default=Path("./ckpt_v7"))

    # ----- 실험 모드 ----- #
    p.add_argument("--exp-mode", choices=["next_year","fixed","walkforward"],
                   default="next_year",
                   help="논문 설계는 next_year 권장")
    # next_year 설정
    p.add_argument("--train-start-year", type=int, default=2017)
    p.add_argument("--train-end-year",   type=int, default=2022,
                   help="학습/튜닝 데이터 종단 연도(포함). 평가연도는 자동으로 +1")
    p.add_argument("--calib-window-months", type=int, default=6,
                   help="튜닝(임계/캘리브레이션)에 사용할 최신 월 수(학습 종단연도 내)")
    # walkforward 설정
    p.add_argument("--wf-stride",  type=int, default=1,   help="WF 오리진 간격(월)")
    p.add_argument("--wf-window",  type=int, default=-1,
                   help="-1=확장창, 양수=슬라이딩창 길이(개월)")
    p.add_argument("--wf-val-window", type=int, default=6,
                   help="WF에서 오리진 직전 검증 월 수(캘리브레이션/임계 추정)")

    # ----- 모델/학습 하이퍼 ----- #
    p.add_argument("--epochs",     type=int,   default=12)
    p.add_argument("--batch-size", type=int,   default=16)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--patience",   type=int,   default=6)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--device",     type=str,   default="cuda:0")
    p.add_argument("--num-workers",type=int,   default=4)

    # ----- 입력 길이/마스크 정책 ----- #
    p.add_argument("--input-len",  type=int,   default=None,
                   help="사용할 과거 AIS 길이 L (기본: dataset_meta.window_len)")
    p.add_argument("--mask-policy", choices=["all","any"], default="all",
                   help="all=모든 horizon 유효샘플만, any=남기되 로스에서 마스크 적용")

    # ----- 멀티-임계 라벨 선택(선택) ----- #
    p.add_argument("--label-suffix", type=str, default="",
                   help="예: 'thr2.5' → Y_1_thr2.5 / Ymask_1_thr2.5 사용")

    # ----- 모델 하이퍼 ----- #
    p.add_argument("--patch-size", type=int, default=24)
    p.add_argument("--embed-ch",   type=int, default=12)
    p.add_argument("--country-emb",type=int, default=16)
    p.add_argument("--static-drop",type=float,default=0.5)

    # ----- 손실 ----- #
    p.add_argument("--loss", choices=["focal","pos_bce"], default="pos_bce")
    p.add_argument("--pos-weight", type=float, default=1.0,
                   help="pos_bce에서 양성가중치. 샘플러와 병용 시 1.0 권장")
    p.add_argument("--focal-gamma",type=float, default=2.0)
    p.add_argument("--focal-alpha",type=float, default=0.30)
    p.add_argument("--h-weights",  type=float, nargs="*", default=None,
                   help="멀티-h 로스 가중치. 미지정 시 균등")

    p.add_argument("--val-metric", choices=["mean_prc","h3_prc","h3_roc"],
                default="mean_prc",
                help="얼리스탑/베스트 스냅샷 선택 지표")

    # ----- 임계/캘리브레이션 ----- #
    p.add_argument("--th-policy",  choices=["f1","youden","budget","cost"], default="budget")
    p.add_argument("--budget",     type=float, default=0.05,
                   help="상위 b 비율 경보(0~1). 논문은 보통 1~10% 범위 보고")
    p.add_argument("--cost-fn",    type=float, default=5.0)
    p.add_argument("--cost-fp",    type=float, default=1.0)
    p.add_argument("--calibration",choices=["none","platt","isotonic"], default="none")

    # ── 새 옵션: 샘플러 기준 horizon / 훈련에서만 국가 필터 ────────────────
    p.add_argument("--sampler-h-ref", type=str, default="auto",
                   help="훈련용 샘플 가중치 계산에 쓸 horizon. 'auto'면 TRAIN 범위에서 양성수가 가장 많은 horizon을 자동 선택")
    p.add_argument("--min-train-pos-per-country", type=int, default=0,
                   help="훈련(TRAIN)에서만 적용: 샘플러 horizon 기준 양성 개수가 이 값보다 작은 국가는 훈련 배제(VAL/TEST는 유지). 0이면 비활성")

    # ----- 저장 ----- #
    p.add_argument("--save-preds",   action="store_true")
    p.add_argument("--save-metrics", action="store_true")
    p.add_argument("--save-calib",   action="store_true")

    return p.parse_args()

# ─────────────────────── 2) Utils ───────────────────────────────────────── #
def seed_everything(seed:int):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def brier_score(y_true, y_prob, mask=None):
    if mask is not None:
        y_true = y_true[mask]; y_prob = y_prob[mask]
    if y_true.size == 0: return np.nan
    return float(np.mean((y_prob - y_true)**2))

def _dedup(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=["win_idx","area_ord","t_end"]).reset_index(drop=True)


def _choose_sampler_h(df_train: pd.DataFrame,
                      labels: Dict[int, np.ndarray],
                      ymasks: Dict[int, np.ndarray],
                      horizons: List[int]) -> tuple[int, int]:
    """TRAIN 구간에서 양성이 가장 많은 horizon을 선택."""
    if len(df_train) == 0:
        return horizons[0], 0
    best_h, best_pos = horizons[0], -1
    a = df_train["area_ord"].to_numpy()
    t = df_train["t_end"].to_numpy()
    for h in horizons:
        y = labels[h][a, t]
        m = ymasks[h][a, t].astype(bool)
        pos = int((y[m] == 1).sum())
        if pos > best_pos:
            best_pos, best_h = pos, h
    return best_h, best_pos

def _filter_train_countries(df_train: pd.DataFrame,
                            labels: Dict[int, np.ndarray],
                            ymasks: Dict[int, np.ndarray],
                            h_ref: int,
                            min_k: int) -> tuple[pd.DataFrame, list[int]]:
    """훈련셋에서만 적용: h_ref 기준 국가별 양성 수 < min_k 인 국가는 제거."""
    if min_k <= 0 or len(df_train) == 0:
        return df_train, sorted(df_train["area_ord"].unique().tolist())
    a = df_train["area_ord"].to_numpy()
    t = df_train["t_end"].to_numpy()
    y = labels[h_ref][a, t]
    m = ymasks[h_ref][a, t].astype(bool)
    is_pos = ((y == 1) & m).astype(np.int32)

    counts = {}
    for area, flag in zip(a, is_pos):
        counts[area] = counts.get(area, 0) + int(flag)

    kept = {area for area, c in counts.items() if c >= min_k}
    if not kept:
        # 모두 날아가면 위험하니 필터 무시
        return df_train, sorted(counts.keys())
    out = df_train[df_train["area_ord"].isin(kept)].reset_index(drop=True)
    return out, sorted(kept)

def ece_score(y_true, y_prob, mask=None, n_bins=10):
    if mask is not None:
        y_true = y_true[mask]; y_prob = y_prob[mask]
    if y_true.size == 0: return np.nan
    bins = np.linspace(0.0, 1.0, n_bins+1)
    idx  = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        sel = (idx==b)
        if not np.any(sel): continue
        conf = y_prob[sel].mean()
        acc  = y_true[sel].mean()
        ece += (sel.mean()) * abs(acc - conf)
    return float(ece)

def youden_threshold(y_true, y_prob):
    if y_true.size == 0: return 0.5
    qs = np.unique(np.quantile(y_prob, np.linspace(0,1,200)))
    best_t, best_j = 0.5, -1
    P = (y_true==1).sum(); N = (y_true==0).sum()
    for t in qs:
        pred = (y_prob >= t)
        tp = (pred & (y_true==1)).sum()
        fn = P - tp
        fp = pred.sum() - tp
        tn = N - fp
        tpr = tp / max((tp+fn),1)
        fpr = fp / max((fp+tn),1)
        j = tpr - fpr
        if j > best_j:
            best_j, best_t = j, t
    return float(best_t)

def f1_threshold(y_true, y_prob):
    if y_true.size == 0: return 0.5
    qs = np.unique(np.quantile(y_prob, np.linspace(0,1,200)))
    best_t, best_f1 = 0.5, -1
    for t in qs:
        pred = (y_prob >= t)
        tp = (pred & (y_true==1)).sum()
        fp = pred.sum() - tp
        fn = (y_true==1).sum() - tp
        f1 = 2*tp / max(2*tp + fp + fn, 1)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)

def cost_threshold(y_true, y_prob, c_fn=5.0, c_fp=1.0):
    if y_true.size == 0: return 0.5
    qs = np.unique(np.quantile(y_prob, np.linspace(0,1,200)))
    best_t, best_cost = 0.5, 1e18
    P = (y_true==1).sum()
    for t in qs:
        pred = (y_prob >= t)
        tp = (pred & (y_true==1)).sum()
        fp = pred.sum() - tp
        fn = P - tp
        cost = c_fn*fn + c_fp*fp
        if cost < best_cost:
            best_cost, best_t = cost, t
    return float(best_t)

def budget_threshold(y_prob, budget=0.05):
    if y_prob.size == 0: return 1.0
    q = np.clip(1.0 - float(budget), 0.0, 1.0)
    return float(np.quantile(y_prob, q))

class _IdentityCalib:
    def fit(self, p, y): return self
    def predict(self, p): return p

class _PlattCalib:
    def __init__(self): self.lr = LogisticRegression(solver="lbfgs")
    def fit(self, p, y):
        if len(np.unique(y)) < 2 or len(y) < 20:
            return self
        p = np.clip(p, 1e-6, 1-1e-6)
        self.lr.fit(p.reshape(-1,1), y)
        return self
    def predict(self, p):
        p = np.clip(p, 1e-6, 1-1e-6)
        try:
            return self.lr.predict_proba(p.reshape(-1,1))[:,1]
        except Exception:
            return p

class _IsoCalib:
    def __init__(self): self.iso = IsotonicRegression(out_of_bounds="clip")
    def fit(self, p, y):
        if len(np.unique(y)) < 2 or len(y) < 20:
            return self
        self.iso.fit(p, y); return self
    def predict(self, p):
        try:   return self.iso.predict(p)
        except Exception: return p

def build_calibrator(name:str):
    if name=="none":     return _IdentityCalib()
    if name=="platt":    return _PlattCalib()
    if name=="isotonic": return _IsoCalib()
    raise ValueError

def month_index_from_meta(meta_json: Path) -> pd.DatetimeIndex:
    meta = json.loads(meta_json.read_text())
    vals = meta.get("time_index", None)
    if vals is not None:
        try:
            return pd.to_datetime(vals, format="%Y%m", utc=True)
        except Exception:
            return pd.to_datetime(vals, utc=True)
    T = int(meta.get("T", 0))
    start_ts = pd.Timestamp("2000-01-01", tz="UTC")
    return pd.date_range(start_ts.normalize(), periods=T, freq="MS", tz="UTC")


def _sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def _git_info():
    try:
        commit = subprocess.check_output(["git","rev-parse","HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        status = subprocess.check_output(["git","status","--porcelain"], stderr=subprocess.DEVNULL).decode().strip()
        dirty  = (status != "")
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return None

def _summarize_split(df: pd.DataFrame, labels, ymasks, horizons: List[int]) -> dict:
    out = {"n": int(len(df))}
    if len(df) == 0:
        for h in horizons: out[f"h{h}_valid"]=0; out[f"h{h}_pos"]=0
        return out
    a = df["area_ord"].to_numpy()
    t = df["t_end"].to_numpy()
    for h in horizons:
        m = ymasks[h][a, t].astype(bool)
        y = labels[h][a, t]
        out[f"h{h}_valid"] = int(m.sum())
        out[f"h{h}_pos"]   = int(((y==1) & m).sum())
    return out
def _dump_manifest(save_dir: Path, args, meta: dict, dates: pd.DatetimeIndex,
                   horizons: List[int], sampler_h: int, model, split_summ: dict,
                   best_epoch: int, best_val: float):
    # 로컬 의존성 (외부 수정 없이 drop-in)
    import sys, platform, os, json
    from datetime import datetime, timezone
    from pathlib import Path as _Path
    import numpy as _np
    import torch as _torch

    # JSON 직렬화 헬퍼
    def _json_default(o):
        if isinstance(o, _Path):
            return str(o)
        if isinstance(o, (set,)):
            return list(o)
        if isinstance(o, (_np.integer,)):
            return int(o)
        if isinstance(o, (_np.floating,)):
            return float(o)
        if isinstance(o, (_np.ndarray,)):
            return o.tolist()
        # torch 관련
        if isinstance(o, _torch.device):
            return str(o)
        if isinstance(o, (_torch.dtype,)):
            return str(o)
        # pandas Timestamp 등
        try:
            import pandas as _pd
            if isinstance(o, _pd.Timestamp):
                return o.isoformat()
        except Exception:
            pass
        # 그 외 알 수 없는 타입은 문자열로
        return str(o)

    # 산출물(artifacts) 스냅샷: 파일명/크기/mtime
    artifacts = []
    try:
        for p in sorted(save_dir.glob("*")):
            if p.is_file():
                stat = p.stat()
                artifacts.append({
                    "name": p.name,
                    "size_bytes": int(stat.st_size),
                    "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
                })
    except Exception:
        pass

    manifest = {
        "cmdline": " ".join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "proc_dir": str(args.proc_dir),
        "label_suffix": args.label_suffix,
        "exp_mode": args.exp_mode,
        "train_years": [int(args.train_start_year), int(args.train_end_year)],
        "test_year": int(args.train_end_year + 1) if args.exp_mode == "next_year" else None,
        "val_metric": getattr(args, "val_metric", "mean_prc"),
        "best_epoch": int(best_epoch),
        "best_val": float(best_val),
        "horizons": [int(h) for h in horizons],
        "sampler_h_ref": int(sampler_h) if sampler_h is not None else None,
        "dataset_meta": {
            "n_countries": meta.get("n_countries", None),
            "time_span": [str(dates.min().date()), str(dates.max().date())],
            "window_len": meta.get("window_len", None),
        },
        "env": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": _torch.__version__,
            "cuda": _torch.version.cuda if _torch.cuda.is_available() else None,
            "gpu": _torch.cuda.get_device_name(0) if _torch.cuda.is_available() else None
        },
        "model": {
            "param_count": int(sum(p.numel() for p in model.parameters())),
            "patch_size": int(args.patch_size),
            "embed_ch": int(args.embed_ch),
            "country_emb": int(args.country_emb),
            "static_drop": float(args.static_drop)
        },
        "train": {
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "batch_size": int(args.batch_size),
            "loss": args.loss,
            "pos_weight": float(args.pos_weight),
            "focal": {"alpha": float(args.focal_alpha), "gamma": float(args.focal_gamma)},
            "h_weights": list(args.h_weights) if args.h_weights is not None else None,
            "mask_policy": args.mask_policy,
            "input_len": int(args.input_len) if args.input_len is not None else None,
            "seed": int(getattr(args, "seed", 0)),
            "sampler_h_ref": getattr(args, "sampler_h_ref", None),
            "min_train_pos_per_country": int(getattr(args, "min_train_pos_per_country", 0))
        },
        "thresholding": {
            "policy": args.th_policy,
            "budget": float(args.budget),
            "cost_fn": float(args.cost_fn),
            "cost_fp": float(args.cost_fp),
            "calibration": args.calibration
        },
        "split_summary": split_summ,
        "artifacts": artifacts,
        "args": vars(args),     # default=_json_default 로 안전 직렬화
        "git": _git_info()      # 기존 함수 그대로 사용
    }

    # 저장 (혹시 또 실패해도 학습 산출물은 남도록 보호)
    try:
        (save_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default))
    except Exception as e:
        try:
            LOG.warning("Failed to write manifest.json: %s", e)
        except Exception:
            print(f"[WARN] Failed to write manifest.json: {e}")

def _json_default(o):
    # json.dumps(..., default=_json_default) 에서 호출됨
    from pathlib import Path
    import numpy as np
    if isinstance(o, Path): return str(o)
    if isinstance(o, (np.integer,)):  return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.ndarray,)):  return o.tolist()
    return str(o)  # 기타 잡다한 타입은 문자열로

# ─────────────────────── 3) Dataset ─────────────────────────────────────── #
# ── dataset/dataloader (drop-in) ────────────────────────────────────────────
class VesselDatasetV7(Dataset):
    def __init__(self, feats, labels, ymasks, sample_df, horizons, phase,
                 device, input_len: Optional[int],mask_policy: str = "all", sampler_h: int | None = None):
        super().__init__()
        self.X  = feats["X"]
        self.S  = feats["S"].float()
        self.M  = feats["M"].float()
        self.Mk = feats["mask_S"].float()
        self.labels = labels
        self.y_masks= ymasks
        self.h_list = horizons
        self.df  = sample_df.reset_index(drop=True)
        self.phase = phase
        self.device = device
        self.Lmax  = self.X.shape[1]
        self.L     = input_len if input_len is not None else self.Lmax
        assert 1 <= self.L <= self.Lmax
        if phase == "train":
            h_ref = sampler_h if sampler_h is not None else self.h_list[0]            
            y_ref = self.labels[h_ref][self.df.area_ord, self.df.t_end]
            m_ref = self.y_masks[h_ref][self.df.area_ord, self.df.t_end].astype(bool)
            pos = (y_ref[m_ref]==1).sum()
            neg = (m_ref.sum() - (y_ref[m_ref]==1).sum())
            w_pos = (neg / max(pos,1))
            valid_any = np.zeros(len(self.df), dtype=bool)
            for h in self.h_list:
                valid_any |= self.y_masks[h][self.df.area_ord, self.df.t_end].astype(bool)
            base_w = np.where(y_ref==1, w_pos, 1.).astype(np.float64)
            if mask_policy == "any":
                base_w = base_w * valid_any.astype(np.float64)
            self.weights = base_w
        else:
            self.weights = None
    def __len__(self): return len(self.df)
    def __getitem__(self, i:int):
        row = self.df.iloc[i]
        widx, cidx, t_end = int(row.win_idx), int(row.area_ord), int(row.t_end)
        X_dyn_full = self.X[widx]
        X_dyn = X_dyn_full[-self.L:]
        S_val  = self.S[cidx, t_end]
        S_mask = self.Mk[cidx, t_end]
        m_rep  = self.M[cidx, t_end]
        y_vec   = torch.tensor([self.labels[h][cidx, t_end] for h in self.h_list], dtype=torch.float32)
        ymask_v = torch.tensor([self.y_masks[h][cidx, t_end] for h in self.h_list], dtype=torch.float32)
        return X_dyn, S_val, S_mask, m_rep, y_vec, ymask_v, torch.tensor(cidx), torch.tensor(t_end)

def build_loader(df: pd.DataFrame, feats, labels, ymasks,
                 horizons, phase, args, sampler_h: int | None = None):
    ds = VesselDatasetV7(feats, labels, ymasks, df, horizons, phase,
                         torch.device(args.device), args.input_len, args.mask_policy,
                         sampler_h=sampler_h)
    if phase == "train":
        w = torch.as_tensor(ds.weights, dtype=torch.double)
        sampler = WeightedRandomSampler(w, num_samples=len(ds), replacement=True)
        return DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                          num_workers=args.num_workers, pin_memory=True, drop_last=False)
    else:
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=max(1, args.num_workers//2),
                          pin_memory=True, drop_last=False)

def _choose_sampler_h(df_train: pd.DataFrame, labels, ymasks, horizons: List[int]) -> Tuple[int, int]:
    best_h, best_pos = horizons[0], -1
    for h in horizons:
        y = labels[h][df_train.area_ord, df_train.t_end]
        m = ymasks[h][df_train.area_ord, df_train.t_end].astype(bool)
        cnt = int((y[m]==1).sum())
        if cnt > best_pos:
            best_pos, best_h = cnt, h
    return best_h, best_pos

def _filter_train_countries(df_train: pd.DataFrame, labels, ymasks,
                            sampler_h: int, min_k: int) -> Tuple[pd.DataFrame, List[int]]:
    if min_k <= 0 or len(df_train)==0:
        return df_train, []
    y = labels[sampler_h][df_train.area_ord, df_train.t_end]
    m = ymasks[sampler_h][df_train.area_ord, df_train.t_end].astype(bool)
    arr_pos = ((y==1) & m)
    tmp = df_train.copy()
    tmp["_pos"] = arr_pos
    pos_by_area = tmp.groupby("area_ord")["_pos"].sum().astype(int)
    keep_areas = pos_by_area[pos_by_area >= min_k].index.astype(int).tolist()
    df_out = df_train[df_train.area_ord.isin(keep_areas)].copy()
    drop_cnt = len(set(df_train.area_ord.unique()) - set(keep_areas))
    LOG.info(f"TRAIN country filter: keep={len(keep_areas)} drop={drop_cnt} (min_k={min_k})")
    return df_out, keep_areas

# ─────────────────────── 4) Loss & Eval ─────────────────────────────────── #
def masked_bce_loss(logits: torch.Tensor, y: torch.Tensor, ymask: torch.Tensor,
                    h_weights: Optional[List[float]], loss_kind: str,
                    pos_weight: float, focal_gamma: float, focal_alpha: float):
    B,H = logits.shape
    if loss_kind == "focal":
        base = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        p_t  = torch.sigmoid(logits)
        p_t  = y * p_t + (1 - y) * (1 - p_t)
        loss = (y * focal_alpha + (1 - y) * (1 - focal_alpha)) \
               * (1 - p_t).pow(focal_gamma) * base
    else:
        pw = torch.tensor([pos_weight], dtype=logits.dtype, device=logits.device)
        loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none",
                                                  pos_weight=pw)

    loss = loss * ymask  # vintage-safe mask
    if h_weights is not None:
        hw = torch.tensor(h_weights, dtype=logits.dtype, device=logits.device).view(1,-1)
        loss = loss * hw
    denom = torch.clamp(ymask.sum(), min=1.0)
    return loss.sum() / denom

@torch.no_grad()
def collect_probs(loader, model, device, horizons, calibrators=None):
    model.eval()
    ys, ms, ps, t_idx, area_idx = [], [], [], [], []
    for batch in tqdm(loader, ncols=80, leave=False, desc="EVAL"):
        X,Sv,Sm,Mv,y,ymask,cidx,t_end = batch
        X      = X.to(device, non_blocking=True)
        Sv     = Sv.to(device, non_blocking=True)
        Sm     = Sm.to(device, non_blocking=True)
        Mv     = Mv.to(device, non_blocking=True)
        cidx   = cidx.to(device, non_blocking=True)

        with torch.amp.autocast('cuda' if device.type=='cuda' else 'cpu', dtype=torch.float16):
            out   = model(X, Sv, Sm, Mv, cidx)
            logits= torch.stack([out[f"h{h}"] for h in horizons], dim=1)
            prob  = torch.sigmoid(logits).detach().cpu().numpy()

        y      = y.detach().cpu().numpy()
        ymask  = ymask.detach().cpu().numpy()
        if calibrators is not None:
            for j,_ in enumerate(horizons):
                prob[:,j] = calibrators[j].predict(prob[:,j])

        ys.append(y); ms.append(ymask); ps.append(prob)
        t_idx.append(t_end.numpy()); area_idx.append(cidx.cpu().numpy())

    ys = np.concatenate(ys,0); ms = np.concatenate(ms,0); ps = np.concatenate(ps,0)
    t_idx = np.concatenate(t_idx,0); area_idx = np.concatenate(area_idx,0)
    return ys, ms, ps, t_idx, area_idx

def compute_table_metrics(y, m, p):
    H = y.shape[1]; out={}
    for j in range(H):
        mask = m[:,j].astype(bool)
        yt, pt = y[:,j][mask], p[:,j][mask]
        if yt.size == 0:
            out[j] = {"roc": np.nan, "prc": np.nan, "brier": np.nan, "ece": np.nan}
            continue
        try:
            roc = roc_auc_score(yt, pt)
        except ValueError:
            roc = np.nan
        prc = average_precision_score(yt, pt) if yt.sum()>0 else np.nan
        brier = brier_score(yt, pt)
        ece   = ece_score(yt, pt)
        out[j] = {"roc":float(roc),"prc":float(prc),"brier":brier,"ece":ece}
    return out

def select_thresholds(val_y, val_m, val_p, horizons, policy, budget, cost_fn, cost_fp):
    ths=[]
    for j,_ in enumerate(horizons):
        mask = val_m[:,j].astype(bool)
        yt, pt = val_y[:,j][mask], val_p[:,j][mask]
        if policy=="f1":      t = f1_threshold(yt, pt)
        elif policy=="youden":t = youden_threshold(yt, pt)
        elif policy=="budget":t = budget_threshold(pt, budget)
        elif policy=="cost":  t = cost_threshold(yt, pt, c_fn=cost_fn, c_fp=cost_fp)
        else: t = 0.5
        ths.append(float(t))
    return ths

def budget_metrics(y, m, p, budget):
    mask = m.astype(bool)
    yt, pt = y[mask], p[mask]
    if yt.size==0: return np.nan, np.nan
    thr = budget_threshold(pt, budget)
    pred = (pt >= thr)
    tp = (pred & (yt==1)).sum()
    pos= (yt==1).sum()
    hit = tp / max(pos,1)
    fa  = (pred & (yt==0)).sum()
    fa_per_100 = fa / (len(yt)/100.0)
    return float(hit), float(fa_per_100)

# ─────────────────────── 5) 공통 로더/라벨 ──────────────────────────────── #
def _suffixize(base:str, suffix:str)->str:
    s = suffix.strip()
    if s == "": return base
    if not s.startswith("thr"): s = f"thr{s}"
    return f"{base}_{s}"

def load_feats_labels(proc_dir: Path, label_suffix: str):
    feat_pt  = torch.load(proc_dir / "features.pt", map_location="cpu", weights_only=False)
    feats = {k: v.contiguous() for k,v in feat_pt.items()}  # X,S,M,mask_S
    meta = json.load(open(proc_dir / "dataset_meta.json"))
    horizons: List[int] = sorted(meta["horizons"])
    n_country: int      = meta["n_countries"]
    month_dim: int      = feats["M"].shape[-1]
    static_dim: int     = feats["S"].shape[-1]
    Lmax: int           = feats["X"].shape[1]
    dates               = month_index_from_meta(proc_dir / "dataset_meta.json")
    df_idx              = pq.read_table(proc_dir / "sample_index.parquet").to_pandas()

    labels_npz = np.load(proc_dir / "labels_all.npz")
    labels = {}
    ymasks = {}
    for h in horizons:
        yk  = _suffixize(f"Y_{h}",     label_suffix)
        mk  = _suffixize(f"Ymask_{h}", label_suffix)
        if yk not in labels_npz or mk not in labels_npz:
            raise KeyError(f"labels_all.npz에 '{yk}' 또는 '{mk}'가 없습니다. (--label-suffix 확인)")
        labels[h] = labels_npz[yk]
        ymasks[h] = labels_npz[mk]

    return feats, meta, horizons, n_country, month_dim, static_dim, Lmax, dates, df_idx, labels, ymasks

def filter_by_mask_policy(df: pd.DataFrame, ymasks: Dict[int, np.ndarray],
                          horizons: List[int], policy: str) -> pd.DataFrame:
    if policy == "all":
        keep = np.ones(len(df), dtype=bool)
        for h in horizons:
            keep &= ymasks[h][df.area_ord, df.t_end].astype(bool)
        return df[keep]
    else:
        return df

# ─────────────────────── 6) Train 루프 ──────────────────────────────────── #
def train_model(loaders, model, args, device):
    loss_kind = "focal" if args.loss=="focal" else "pos_bce"
    opt   = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    try:
        scaler = torch.amp.GradScaler('cuda')
    except Exception:
        scaler = torch.cuda.amp.GradScaler()

    #best_val, best_state, stale = -1e18, None, 0
    best_val, best_state, best_epoch, stale = -1e18, None, -1, 0
    h_weights = args.h_weights
    horizons  = model.horizons
    hist = []

    for ep in range(1, args.epochs+1):
        # ---- train ----
        model.train()
        loop = tqdm(loaders["train"], ncols=80, leave=False, desc=f"TRAIN {ep:02d}")
        run_loss, steps = 0.0, 0
        for batch in loop:
            X,Sv,Sm,Mv,y,ymask,cidx,_ = batch
            X,Sv,Sm,Mv,cidx,y,ymask = (X.to(device), Sv.to(device), Sm.to(device),
                                       Mv.to(device), cidx.to(device),
                                       y.to(device), ymask.to(device))
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda' if device.type=='cuda' else 'cpu', dtype=torch.float16):
                out   = model(X, Sv, Sm, Mv, cidx)
                logits= torch.stack([out[f"h{h}"] for h in horizons], dim=1)
                loss  = masked_bce_loss(logits, y, ymask, h_weights, loss_kind,
                                        args.pos_weight, args.focal_gamma, args.focal_alpha)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            loop.set_postfix(loss=f"{loss.item():.4f}")
            run_loss += float(loss.item()); steps += 1
        sched.step()

        # ---- val ----
        yv, mv, pv, _, _ = collect_probs(loaders["val"], model, device, horizons)

        mets = compute_table_metrics(yv, mv, pv)
        # 선택 지표 산출
        if args.val_metric == "mean_prc":
            score = np.nanmean([m["prc"] for m in mets.values()])
        elif args.val_metric == "h3_prc":
            j = horizons.index(3) if 3 in horizons else 0
            score = mets[j]["prc"]
        else:  # "h3_roc"
            j = horizons.index(3) if 3 in horizons else 0
            score = mets[j]["roc"]
        LOG.info(f"[E{ep:03d}] val({args.val_metric})={score:.3f} | per-h AUPRC: "
                 + " ".join([f"h{h}:{mets[i]['prc']:.3f}" for i,h in enumerate(horizons)]))
        # 히스토리 기록
        h3_idx = horizons.index(3) if 3 in horizons else 0
        hist.append({
            "epoch": ep,
            "train_loss": (run_loss/steps) if steps>0 else None,
            "val_mean_prc": float(np.nanmean([m["prc"] for m in mets.values()])),
            "val_h3_prc": float(mets[h3_idx]["prc"]),
            "val_h3_roc": float(mets[h3_idx]["roc"])
        })
        if (best_state is None) or (score >= best_val):
            best_val, best_state, best_epoch, stale = score, {k:v.cpu() for k,v in model.state_dict().items()}, ep, 0



        else:
            stale += 1
            if stale >= args.patience:
                LOG.info("Early-stopping"); break

    model.load_state_dict(best_state)
    return model, hist, best_epoch, best_val

# ─────────────────────── 7) 모드: next_year ─────────────────────────────── #
def run_next_year(args):
    device = torch.device(args.device)
    seed_everything(args.seed)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    feats, meta, horizons, n_country, month_dim, static_dim, Lmax, dates, df_idx, labels, ymasks = \
        load_feats_labels(args.proc_dir, args.label_suffix)
    if args.input_len is None: args.input_len = Lmax

    # 연도 마스킹
    y0, y1 = args.train_start_year, args.train_end_year
    y_test = y1 + 1
    if (dates.year.min() > y0) or (dates.year.max() < y_test):
        LOG.warning("요청 연도가 데이터 범위를 벗어날 수 있습니다: %s~%s vs dataset %s~%s",
                    y0, y_test, dates.year.min(), dates.year.max())

    # t_end 인덱스
    t_train = np.where((dates.year >= y0) & (dates.year <= y1))[0]
    t_test  = np.where(dates.year == y_test)[0]
    if len(t_test)==0:
        raise SystemExit(f"평가연도 {y_test} 가 데이터에 없습니다.")

    # 학습/튜닝에서 라벨 누수 금지: t_end + h_max ≤ max(t_train) 조건
    h_max = max(horizons)
    t_train_cut = t_train[t_train + h_max <= t_train.max()]
    # 튜닝(임계/캘리브레이션)용: 학습 종단연도 내 마지막 k개월
    k = int(args.calib_window_months)
    last_months = sorted(t_train_cut[t_train_cut >= (t_train_cut.max() - (k-1))])

    # DataFrame 슬라이싱
    def slice_df(ts: np.ndarray) -> pd.DataFrame:
        sub = df_idx[df_idx.t_end.isin(ts)]
        return filter_by_mask_policy(sub, ymasks, horizons, args.mask_policy)

    df_train = slice_df(t_train_cut)
    df_val   = slice_df(np.array(last_months, dtype=int))
    df_test  = slice_df(t_test)

    df_train = _dedup(df_train)
    df_val   = _dedup(df_val)
    df_test  = _dedup(df_test)

    # 스플릿 저장
    pq.write_table(pa.Table.from_pandas(df_train), args.save_dir / "split_train.parquet")
    pq.write_table(pa.Table.from_pandas(df_val),   args.save_dir / "split_val.parquet")
    pq.write_table(pa.Table.from_pandas(df_test),  args.save_dir / "split_test.parquet")
    split_summ = {
        "train": _summarize_split(df_train, labels, ymasks, horizons),
        "val":   _summarize_split(df_val,   labels, ymasks, horizons),
        "test":  _summarize_split(df_test,  labels, ymasks, horizons),
    }
    (args.save_dir / "split_summary.json").write_text(json.dumps(split_summ, indent=2))

    # 1) 샘플러 horizon 결정
    if str(args.sampler_h_ref).strip().lower() == "auto":
        sampler_h, pos_cnt = _choose_sampler_h(df_train, labels, ymasks, horizons)
        LOG.info(f"[NEXT] sampler auto → h{sampler_h} (train pos={pos_cnt})")
    else:
        try:
            sampler_h = int(args.sampler_h_ref)
            if sampler_h not in horizons: raise ValueError
        except Exception:
            sampler_h, pos_cnt = _choose_sampler_h(df_train, labels, ymasks, horizons)
            LOG.warning(f"[NEXT] invalid --sampler-h-ref → auto fallback: h{sampler_h}")

    # 2) TRAIN 국가 필터(훈련 데이터에만 적용)
    if args.min_train_pos_per_country and args.min_train_pos_per_country > 0:
        df_train, kept_areas = _filter_train_countries(
            df_train, labels, ymasks, sampler_h, args.min_train_pos_per_country
        )
        LOG.info(f"[NEXT] TRAIN country filter: keep={len(kept_areas)} (min_k={args.min_train_pos_per_country})")

    loaders = {
        "train": build_loader(df_train, feats, labels, ymasks, horizons, "train", args, sampler_h),
        "val":   build_loader(df_val, feats, labels, ymasks, horizons, "val",   args),
        "test":  build_loader(df_test,  feats, labels, ymasks, horizons, "val",   args),
    }

    # 모델
    _, _, C, H, W = feats["X"].shape
    model = build_model_v7(
        in_ch=C, img_h=H, img_w=W,
        d_s_val=static_dim, d_s_mask=static_dim,
        month_dim=month_dim, n_country=n_country,
        horizons=horizons,
        patch_size=args.patch_size, embed_ch=args.embed_ch,
        country_emb_dim=args.country_emb,
        dropout_static=args.static_drop,
        device=device
    )
    LOG.info(f"Model params = {sum(p.numel() for p in model.parameters()):,}")

    # 학습
    #model = train_model(loaders, model, args, device)
    model, hist, best_ep, best_val = train_model(loaders, model, args, device)
    # 트레이닝 로그 저장
    pd.DataFrame(hist).to_csv(args.save_dir / "train_log.csv", index=False)

    # 캘리브레이션/임계(튜닝 윈도우에서)
    #val_y, val_m, val_p, _, _ = collect_probs(loaders["val"], model, device, horizons)
    val_y, val_m, val_p, val_t, val_a = collect_probs(loaders["val"], model, device, horizons)
    
    calibrators=[]
    for j,_ in enumerate(horizons):
        cal = build_calibrator(args.calibration)
        mask = val_m[:,j].astype(bool)
        calibrators.append(cal.fit(val_p[:,j][mask], val_y[:,j][mask]))

    # 테스트
    #test_y, test_m, test_p, t_end, a_ord = collect_probs(loaders["test"], model, device, horizons, calibrators)
    test_y, test_m, test_p, t_end, a_ord = collect_probs(loaders["test"], model, device, horizons, calibrators)    
    table = compute_table_metrics(test_y, test_m, test_p)

    # 예산 기반
    for j,h in enumerate(horizons):
        hit, fa100 = budget_metrics(test_y[:,j], test_m[:,j], test_p[:,j], args.budget)
        table[j]["hit@b"] = hit; table[j]["fa@b"] = fa100
    LOG.info("TEST per-h metrics: " + " | ".join([
        f"h{h}: AUPRC={table[j]['prc']:.3f} AUROC={table[j]['roc']:.3f} "
        f"Brier={table[j]['brier']:.3f} ECE={table[j]['ece']:.3f} Hit@b={table[j]['hit@b']:.3f}"
        for j,h in enumerate(horizons)
    ]))

    # 월별 OOS 통계(테스트 연도)
    test_dates = dates[t_end]
    per_month=[]
    for j,h in enumerate(horizons):
        for d in np.unique(test_dates):
            sel = (test_dates==d) & test_m[:,j].astype(bool)
            yt = test_y[:,j][sel]; pt = test_p[:,j][sel]
            if yt.size==0:
                roc=prc=brier=ece=np.nan
            else:
                try: roc = roc_auc_score(yt, pt)
                except: roc = np.nan
                prc = average_precision_score(yt, pt) if yt.sum()>0 else np.nan
                brier = brier_score(yt, pt); ece = ece_score(yt, pt)
            per_month.append({"date":str(d.date()), "horizon":h,
                              "n":int(sel.sum()), "pos":int(yt.sum() if yt.size else 0),
                              "auroc":roc, "auprc":prc, "brier":brier, "ece":ece})
    df_month = pd.DataFrame(per_month)

    # 저장
    args.save_dir.mkdir(parents=True, exist_ok=True)
    #torch.save(model.state_dict(), args.save_dir / "best_v7.pt")
    # 체크포인트(전체)
    torch.save({
        "state_dict": model.state_dict(),
        "args": vars(args),
        "horizons": horizons,
        "sampler_h_ref": sampler_h,
        "best_epoch": best_ep,
        "best_val": best_val
    }, args.save_dir / "checkpoint.pt")
    # 가벼운 state_dict도 유지
    torch.save(model.state_dict(), args.save_dir / "best_v7.pt")

    if args.save_preds:
        recs=[]
        for j,h in enumerate(horizons):
            for i in range(len(test_p)):
                recs.append({
                    "date": str(test_dates[i].date()),
                    "area_ord": int(a_ord[i]),
                    "horizon": h,
                    "y_true": int(test_y[i, j]),
                    "y_mask": int(test_m[i, j]),
                    "y_prob": float(test_p[i, j]),
                    "split": "test"
                })
        pq.write_table(pa.Table.from_pandas(pd.DataFrame(recs)),
                       args.save_dir / "preds_next_year.parquet")

        # 검증 예측도 저장
        recs_val=[]
        for j,h in enumerate(horizons):
            for i in range(len(val_p)):
                recs_val.append({
                    "date": str(dates[val_t[i]].date()),
                    "area_ord": int(val_a[i]),
                    "horizon": h,
                    "y_true": int(val_y[i, j]),
                    "y_mask": int(val_m[i, j]),
                    "y_prob": float(val_p[i, j]),
                    "split": "val"
                })
        pq.write_table(pa.Table.from_pandas(pd.DataFrame(recs_val)),
                       args.save_dir / "preds_next_year_val.parquet")


    if args.save_metrics:
        rows=[]
        for j,h in enumerate(horizons):
            rows.append({"horizon":h, **table[j]})
        pd.DataFrame(rows).to_csv(args.save_dir / "metrics_next_year.csv", index=False)
        df_month.to_csv(args.save_dir / "metrics_next_year_by_month.csv", index=False)

        # 국가별 지표
        per_cty=[]
        for j,h in enumerate(horizons):
            for c in np.unique(a_ord):
                sel = (a_ord==c) & test_m[:,j].astype(bool)
                yt = test_y[:,j][sel]; pt = test_p[:,j][sel]
                if yt.size==0 or yt.sum()==0 or (yt==0).all():
                    roc=prc=np.nan
                else:
                    try: roc = roc_auc_score(yt, pt)
                    except: roc = np.nan
                    prc = average_precision_score(yt, pt) if yt.sum()>0 else np.nan
                per_cty.append({"area_ord":int(c), "horizon":h, "n":int(sel.sum()),
                                "pos":int(yt.sum() if yt.size else 0),
                                "auroc":roc, "auprc":prc})
        pd.DataFrame(per_cty).to_csv(args.save_dir / "metrics_next_year_by_country.csv", index=False)
        # ROC/PR 곡선 저장
        for j,h in enumerate(horizons):
            sel = test_m[:,j].astype(bool)
            yt = test_y[:,j][sel]; pt = test_p[:,j][sel]
            if yt.size>0 and ((yt==0).any() and (yt==1).any()):
                fpr,tpr,_ = roc_curve(yt, pt)
                pd.DataFrame({"fpr":fpr,"tpr":tpr}).to_csv(args.save_dir / f"roc_curve_h{h}.csv", index=False)
                pre,rec,_ = precision_recall_curve(yt, pt)
                pd.DataFrame({"recall":rec,"precision":pre}).to_csv(args.save_dir / f"pr_curve_h{h}.csv", index=False)
        # Budget 혼동표/Hit 요약(참고)
        bsum=[]
        for j,h in enumerate(horizons):
            sel = test_m[:,j].astype(bool); yt = test_y[:,j][sel]; pt = test_p[:,j][sel]
            if yt.size>0:
                thr = budget_threshold(pt, args.budget)
                pred = (pt >= thr).astype(int)
                tp = int(((pred==1)&(yt==1)).sum()); fp=int(((pred==1)&(yt==0)).sum())
                fn = int(((pred==0)&(yt==1)).sum()); tn=int(((pred==0)&(yt==0)).sum())
                bsum.append({"horizon":h, "thr":float(thr), "tp":tp, "fp":fp, "fn":fn, "tn":tn})
        pd.DataFrame(bsum).to_csv(args.save_dir / "metrics_next_year_budget.csv", index=False)


    if args.save_calib and args.calibration!="none":
        import pickle
        with open(args.save_dir / "calibrators.pkl","wb") as f:
            pickle.dump(calibrators, f)


    # 코드 스냅샷 & 매니페스트
    snap_dir = args.save_dir / "code_snapshot"; snap_dir.mkdir(exist_ok=True)
    this_py  = Path(__file__).resolve()
    model_py = (Path(__file__).parent / "model_v7.py").resolve()
    shutil.copy2(this_py,  snap_dir / "train_v7.py")
    shutil.copy2(model_py, snap_dir / "model_v7.py")
    (args.save_dir / "sha256_train_v7.txt").write_text(_sha256sum(this_py))
    (args.save_dir / "sha256_model_v7.txt").write_text(_sha256sum(model_py))
    _dump_manifest(args.save_dir, args, meta, dates, horizons, sampler_h, model,
                   split_summ, best_ep, best_val)

# ─────────────────────── 8) 모드: fixed & walkforward ───────────────────── #
def run_fixed(args):
    device = torch.device(args.device)
    seed_everything(args.seed)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    feats, meta, horizons, n_country, month_dim, static_dim, Lmax, dates, df_idx, labels, ymasks = \
        load_feats_labels(args.proc_dir, args.label_suffix)
    if args.input_len is None: args.input_len = Lmax

    splits   = torch.load(args.proc_dir / "splits.pt", map_location="cpu", weights_only=False)
    def mask_split(name:str):
        idx_bool = splits[name]  # (T,)
        keep = df_idx.t_end.map(lambda t: bool(idx_bool[t]))
        return filter_by_mask_policy(df_idx[keep], ymasks, horizons, args.mask_policy)

    # --- split 데이터프레임 준비
    df_train = mask_split("train")
    df_val   = mask_split("val")
    df_test  = mask_split("test")

    df_train = _dedup(df_train)
    df_val   = _dedup(df_val)
    df_test  = _dedup(df_test)


    # --- 샘플러 horizon 선택
    if args.sampler_h_ref.strip().lower() == "auto":
        sampler_h, pos_cnt = _choose_sampler_h(df_train, labels, ymasks, horizons)
        LOG.info(f"[sampler] auto → h{sampler_h} (train positives={pos_cnt})")
    else:
        try:
            sampler_h = int(args.sampler_h_ref)
            if sampler_h not in horizons:
                raise ValueError
        except Exception:
            sampler_h, pos_cnt = _choose_sampler_h(df_train, labels, ymasks, horizons)
            LOG.warning(f"[sampler] invalid --sampler-h-ref → auto fallback: h{sampler_h}")

    # --- TRAIN 국가 필터(선택)
    df_train, kept_areas = _filter_train_countries(
        df_train, labels, ymasks, sampler_h, args.min_train_pos_per_country
    )

    loaders = {
        "train": build_loader(df_train, feats, labels, ymasks, horizons, "train", args, sampler_h),
        "val":   build_loader(df_val,   feats, labels, ymasks, horizons, "val",   args),
        "test":  build_loader(df_test,  feats, labels, ymasks, horizons, "test",  args),
    }

    _, _, C, H, W = feats["X"].shape
    model = build_model_v7(
        in_ch=C, img_h=H, img_w=W,
        d_s_val=static_dim, d_s_mask=static_dim,
        month_dim=month_dim, n_country=n_country,
        horizons=horizons,
        patch_size=args.patch_size, embed_ch=args.embed_ch,
        country_emb_dim=args.country_emb,
        dropout_static=args.static_drop,
        device=device
    )
    LOG.info(f"Model params = {sum(p.numel() for p in model.parameters()):,}")

    model = train_model(loaders, model, args, device)

    # 캘리브레이션/임계
    val_y, val_m, val_p, _, _ = collect_probs(loaders["val"], model, device, horizons)
    calibrators=[]
    for j,_ in enumerate(horizons):
        cal = build_calibrator(args.calibration)
        mask = val_m[:,j].astype(bool)
        calibrators.append(cal.fit(val_p[:,j][mask], val_y[:,j][mask]))

    test_y, test_m, test_p, t_end, a_ord = collect_probs(loaders["test"], model, device, horizons, calibrators)
    table = compute_table_metrics(test_y, test_m, test_p)
    for j,h in enumerate(horizons):
        hit, fa100 = budget_metrics(test_y[:,j], test_m[:,j], test_p[:,j], args.budget)
        table[j]["hit@b"] = hit; table[j]["fa@b"] = fa100
    LOG.info("TEST per-h metrics: " + " | ".join([
        f"h{h}: AUPRC={table[j]['prc']:.3f} AUROC={table[j]['roc']:.3f} "
        f"Brier={table[j]['brier']:.3f} ECE={table[j]['ece']:.3f} Hit@b={table[j]['hit@b']:.3f}"
        for j,h in enumerate(horizons)
    ]))

    # 저장
    torch.save(model.state_dict(), args.save_dir / "best_v7.pt")
    if args.save_preds:
        test_dates = dates[t_end]
        recs=[]
        for j,h in enumerate(horizons):
            for i in range(len(test_p)):
                recs.append({
                    "date": str(test_dates[i].date()),
                    "area_ord": int(a_ord[i]),
                    "horizon": h,
                    "y_true": int(test_y[i, j]),
                    "y_mask": int(test_m[i, j]),
                    "y_prob": float(test_p[i, j]),
                    "split": "test"
                })
        pq.write_table(pa.Table.from_pandas(pd.DataFrame(recs)),
                       args.save_dir / "preds_fixed.parquet")
    if args.save_metrics:
        rows=[]
        for j,h in enumerate(horizons):
            rows.append({"horizon":h, **table[j]})
        pd.DataFrame(rows).to_csv(args.save_dir / "metrics_fixed.csv", index=False)

def run_walkforward(args):
    device = torch.device(args.device)
    seed_everything(args.seed)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    feats, meta, horizons, n_country, month_dim, static_dim, Lmax, dates, df_idx, labels, ymasks = \
        load_feats_labels(args.proc_dir, args.label_suffix)
    if args.input_len is None: args.input_len = Lmax

    splits   = torch.load(args.proc_dir / "splits.pt", map_location="cpu", weights_only=False)
    test_mask = splits["test"].astype(bool)
    origin_ts = np.where(test_mask)[0]
    origin_ts = origin_ts[::max(1,args.wf_stride)]

    preds_rows = []
    month_rows = []
    h_max = max(horizons)

    for t0 in origin_ts:
        LOG.info(f"WF origin t0={str(dates[t0].date())} …")

        if args.wf_window < 0: train_start = 0
        else:                   train_start = max(0, t0 - args.wf_window + 1)
        train_end = t0
        val_start = max(0, t0 - args.wf_val_window + 1)
        val_end   = t0

        def pick(df, a, b, require_past=True):
            sel = (df.t_end >= a) & (df.t_end <= b)
            if require_past: sel &= (df.t_end + h_max <= t0)
            return df[sel]
        
        df_train = filter_by_mask_policy(pick(df_idx, train_start, train_end, True), ymasks, horizons, args.mask_policy)
        df_val   = filter_by_mask_policy(pick(df_idx, val_start,   val_end,   True), ymasks, horizons, args.mask_policy)

        df_train = _dedup(df_train)
        df_val   = _dedup(df_val)


        # ★ 추가 1) 샘플러 horizon 선택 (auto 또는 정수 지정)
        if str(args.sampler_h_ref).strip().lower() == "auto":
            sampler_h, pos_cnt = _choose_sampler_h(df_train, labels, ymasks, horizons)
            LOG.info(f"[WF {str(dates[t0].date())}] sampler auto → h{sampler_h} (train pos={pos_cnt})")
        else:
            try:
                sampler_h = int(args.sampler_h_ref)
                if sampler_h not in horizons: raise ValueError
            except Exception:
                sampler_h, pos_cnt = _choose_sampler_h(df_train, labels, ymasks, horizons)
                LOG.warning(f"[WF {str(dates[t0].date())}] invalid --sampler-h-ref → auto fallback: h{sampler_h}")

        # ★ 추가 2) TRAIN 국가 필터링 (훈련에서만 제외; VAL/TEST는 그대로)
        if args.min_train_pos_per_country and args.min_train_pos_per_country > 0:
            df_train, kept_areas = _filter_train_countries(
                df_train, labels, ymasks, sampler_h, args.min_train_pos_per_country
            )
            LOG.info(f"[WF {str(dates[t0].date())}] TRAIN country filter: keep={len(kept_areas)} (min_k={args.min_train_pos_per_country})")

        # ★ 추가 3) train 로더에 sampler_h 전달
        loaders = {
            "train": build_loader(df_train, feats, labels, ymasks, horizons, "train", args, sampler_h),
            "val":   build_loader(df_val,   feats, labels, ymasks, horizons, "val",   args),
        }

        _, _, C, H, W = feats["X"].shape
        model = build_model_v7(
            in_ch=C, img_h=H, img_w=W,
            d_s_val=static_dim, d_s_mask=static_dim,
            month_dim=month_dim, n_country=n_country,
            horizons=horizons,
            patch_size=args.patch_size, embed_ch=args.embed_ch,
            country_emb_dim=args.country_emb,
            dropout_static=args.static_drop,
            device=device
        )
        model = train_model(loaders, model, args, device)

        # 오리진 직전 val로 캘리브레이션
        val_y, val_m, val_p, _, _ = collect_probs(loaders["val"], model, device, horizons)
        calibrators=[]
        for j,_ in enumerate(horizons):
            cal = build_calibrator(args.calibration)
            mask = val_m[:,j].astype(bool)
            calibrators.append(cal.fit(val_p[:,j][mask], val_y[:,j][mask]))

        # 오리진 월 OOS
        df_t0 = df_idx[df_idx.t_end == t0]
        loader_t0 = build_loader(filter_by_mask_policy(df_t0, ymasks, horizons, args.mask_policy),
                                 feats, labels, ymasks, horizons, "val", args)
        y_t0, m_t0, p_t0, t_end_t0, a_ord_t0 = collect_probs(loader_t0, model, device, horizons, calibrators)

        for j,h in enumerate(horizons):
            for i in range(len(p_t0)):
                preds_rows.append({
                    "origin": str(dates[t0].date()),
                    "date":   str(dates[t_end_t0[i]].date()),
                    "area_ord": int(a_ord_t0[i]),
                    "horizon": h,
                    "y_true": int(y_t0[i, j]),
                    "y_mask": int(m_t0[i, j]),
                    "y_prob": float(p_t0[i, j]),
                    "split":  "wf"
                })
        # 월별 지표
        for j,h in enumerate(horizons):
            sel = m_t0[:,j].astype(bool)
            yt = y_t0[:,j][sel]; pt = p_t0[:,j][sel]
            if yt.size==0:
                roc=prc=brier=ece=np.nan; hit=fa100=np.nan
            else:
                try: roc = roc_auc_score(yt, pt)
                except: roc = np.nan
                prc = average_precision_score(yt, pt) if yt.sum()>0 else np.nan
                brier = brier_score(yt, pt); ece = ece_score(yt, pt)
                hit, fa100 = budget_metrics(yt, np.ones_like(yt,dtype=bool), pt, args.budget)
            month_rows.append({"origin":str(dates[t0].date()), "horizon":h,
                               "auroc":roc, "auprc":prc, "brier":brier, "ece":ece,
                               "hit@b":hit, "fa@b":fa100, "n":int(sel.sum()),
                               "pos":int(yt.sum() if yt.size else 0)})

    if args.save_preds and len(preds_rows)>0:
        pq.write_table(pa.Table.from_pandas(pd.DataFrame(preds_rows)),
                       args.save_dir / "preds_wf.parquet")
    if args.save_metrics and len(month_rows)>0:
        pd.DataFrame(month_rows).to_csv(args.save_dir / "metrics_wf_by_origin.csv", index=False)

# ─────────────────────── 9) Main ────────────────────────────────────────── #
def main():
    args = parse_args()
    if args.h_weights is not None:
        LOG.info(f"h-weights={args.h_weights}")

    # 파일 로그
    args.save_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(args.save_dir / "train.log")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(fh)

    if args.exp_mode == "next_year":
        run_next_year(args)
    elif args.exp_mode == "fixed":
        run_fixed(args)
    else:
        run_walkforward(args)

if __name__ == "__main__":
    main()
