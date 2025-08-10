#!/usr/bin/env python3
"""
emodnet_prepare_tensor_v4.py  – 2025‑07‑27  (rev‑B)

주요 특징
────────
1. 육지 셀 → sentinel(‑1)  /  조용바다 0  /  선박 > 0
2. land_mask · active_mask 채널(옵션) → C+2
3. 활성 해역 기준 z‑score (float32 누적, float16 캐스팅)
4. PyTorch 2 sparse COO 저장 (--sparse-out)
5. 메타 정보: channels, mean_active, std_active, sentinel 포함

python emodnet_prepare_tensor_v4.py   --root /home/skmoon/DB/emodnet/emodnet   --start 201701 --end 202312   --ship-types 09 10 all   --bbox 27 40 42 47   --log1p --normalise   --dtype float16 --layout TCHW   --missing nan --add-masks   --sentinel -20   --out ~/data/blacksea_tensor_log1p.npz   --sparse-out ~/data/blacksea_sparse.pt

"""

from __future__ import annotations
import argparse, calendar, json, logging, re, sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
import torch                     # sparse 저장 시 필요
from pyproj import Transformer
from rasterio.transform import Affine
from rasterio.windows import from_bounds

LOG = logging.getLogger("emodnet")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")

DEF_BBOX = (27.0, 40.0, 42.0, 47.0)  # Black Sea

# ───────────────────────────────── month iterator
def month_iter(start: str, end: str):
    cur = datetime.strptime(start, "%Y%m")
    end_dt = datetime.strptime(end, "%Y%m")
    while cur <= end_dt:
        yield cur.year, cur.month
        cur = cur.replace(year=cur.year + (cur.month // 12),
                          month=(cur.month % 12) + 1)

# ───────────────────────────────── build index
def build_index(root: Path, recursive: bool, glob: Optional[str]):
    idx: Dict[Tuple[str, str], Path] = {}
    paths = (root.rglob(glob) if recursive else root.glob(glob)) if glob \
            else (root.rglob("*.tif") if recursive else root.glob("*.tif"))
    pat = re.compile(r"vesseldensity_(?P<code>[0-9a-z]+)_(?P<yyyymm>\d{6})", re.I)
    for p in paths:
        m = pat.match(p.stem) if not glob else None
        code, yyyymm = (m.group("code"), m.group("yyyymm")) if m else p.stem.split("_")[1:3]
        idx[(code.lower(), yyyymm)] = p
    return idx

# ───────────────────────────────── argparse
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--ship-types", nargs="*", default=["09", "10", "all"])
    ap.add_argument("--bbox", nargs=4, type=float, default=DEF_BBOX)
    ap.add_argument("--layout", choices=["THWC", "TCHW"], default="THWC")
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float16")
    ap.add_argument("--normalise", action="store_true")
    ap.add_argument("--log1p", action="store_true")
    ap.add_argument("--missing", choices=["skip", "nan", "error"], default="error")
    ap.add_argument("--recursive", action="store_true", default=True)
    ap.add_argument("--no-recursive", dest="recursive", action="store_false")
    ap.add_argument("--strict-glob")
    ap.add_argument("--add-masks", action="store_true")
    ap.add_argument("--sentinel", type=float, default=-20.0)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sparse-out", type=Path)
    return ap.parse_args()

# ─────────────────────────────────── main
def main():
    a = get_args()
    months = list(month_iter(a.start, a.end))
    ship_types = [c.lower() for c in a.ship_types]
    T, C_core = len(months), len(ship_types)

    # bbox -> 3035
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    minx, miny = tr.transform(a.bbox[0], a.bbox[1])
    maxx, maxy = tr.transform(a.bbox[2], a.bbox[3])
    LOG.info("BBox 3035: %.0f %.0f %.0f %.0f", minx, miny, maxx, maxy)

    idx = build_index(a.root, a.recursive, a.strict_glob)
    if not idx:
        LOG.error("No GeoTIFFs in %s", a.root); sys.exit(1)

    tensor = None
    affine_ref = crs_ref = win = None
    time_index: List[str] = []

    for t, (y, m) in enumerate(months):
        yyyymm = f"{y}{m:02d}"; time_index.append(yyyymm)
        for c, code in enumerate(ship_types):
            path = idx.get((code, yyyymm))


            if path is None:
                if a.missing == "error":
                    LOG.error("Missing %s %s", code, yyyymm); sys.exit(1)
                elif a.missing == "skip":
                    continue
                else:                                      # --missing nan
                   if tensor is not None:                 # 이미 텐서가 있으면
                       tensor[t, :, :, c] = a.sentinel    # ★ sentinel로 채움
                   continue


            with rasterio.open(path) as src:
                if tensor is None:
                    win = from_bounds(minx, miny, maxx, maxy, src.transform) \
                          .round_offsets().round_lengths()
                    H, W = int(win.height), int(win.width)
                    extra = 2 if a.add_masks else 0
                    dt = np.float32 if a.dtype == "float32" else np.float16
                    tensor = np.full((T, H, W, C_core+extra), np.nan, dt)
                    affine_ref, crs_ref = src.transform, src.crs
                    LOG.info("Tensor %s", tensor.shape)

                band = src.read(1, window=win, masked=True).astype(np.float32)
                band = np.clip(band, 0, 1e7)

                if a.normalise:
                    band /= 24 * calendar.monthrange(y, m)[1]
                if a.log1p:
                    band = np.log1p(band)              # 이미 0~1e6 범위

                land_mask = band.mask
                band = band.filled(np.nan)
                band[np.isnan(band)] = a.sentinel
                active_mask = band > 0

                tensor[t, :, :, c] = band.astype(tensor.dtype, copy=False)
                if a.add_masks and c == 0:
                    tensor[t, :, :, C_core]   = land_mask.astype(tensor.dtype)
                    tensor[t, :, :, C_core+1] = active_mask.astype(tensor.dtype)

    if tensor is None:
        LOG.error("No rasters loaded"); sys.exit(1)

    # z‑score (float32 누적)

    core_f32 = tensor[..., :C_core].astype(np.float32, copy=True)
    # ★ 원본 값이 0보다 큰 위치를 미리 기억 (sparse 저장용)

    # 채널축은 항상 마지막(-1) → layout 무관
    positive_mask = (core_f32 > 0).any(axis=-1)         # (T,H,W)

    # (land·quiet 제외 + 채널별 활성 픽셀만)
    """
    active_mask4d = np.broadcast_to(
        positive_mask[..., None],   # (T,H,W,1)
        core_f32.shape)             # (T,H,W,C_core)
    """
    active_mask4d = core_f32 > 0           # 채널별 활성 위치 (sparse 용)
    # ─────────────── 스케일링은 DataLoader 단계에서 수행 ───────────────
    # 여기서는 값을 수정하지 않고, μ·σ 는 placeholder 로 남겨 메타만 채웁니다.
    mu  = np.nan
    std = np.nan
    if a.layout == "TCHW":
        tensor = np.transpose(tensor, (0, 3, 1, 2))
        #positive_mask = np.transpose(positive_mask, (0, 3, 1, 2))  # (T,C,H,W)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, X=tensor)
    LOG.info("Dense NPZ → %s (%.1f MiB)", a.out, tensor.nbytes/1024**2)

    # sparse 저장
    if a.sparse_out:
        X = torch.from_numpy(tensor)
        # positive_mask → torch.BoolTensor 로 변환

        pos_mask_pt = torch.from_numpy(positive_mask)        # (T,H,W) Bool
        # ➊ 채널 차원을 포함해 4‑D 좌표로 저장  (T,C,H,W) 또는 (T,H,W,C)
        if a.layout == "TCHW":
            coords_4d = torch.nonzero(
                (X[:, :C_core] > 0), as_tuple=False).t()   # (4,M)
            vals = X[coords_4d[0], coords_4d[1],
                     coords_4d[2], coords_4d[3]].to(torch.float16)
        else:  # THWC
            coords_4d = torch.nonzero(
                (X[..., :C_core] > 0), as_tuple=False).t()  # (4,M)
            vals = X[coords_4d[0], coords_4d[1],
                     coords_4d[2], coords_4d[3]].to(torch.float16)
        coords = coords_4d  # 이름 유지
        
        
        vals = vals.to(torch.float16)

        torch.save({"coords": coords,
                    "values": vals,
                    "shape": X.shape,
                    "mean": float(mu),
                    "std":  float(std)},
                   a.sparse_out)
        LOG.info("Sparse PT → %s (%.1f MiB)", a.sparse_out,
                 vals.numel()*2/1024**2)

    channels = ship_types + (["land", "active"] if a.add_masks else [])
    meta = dict(
        bbox_ll=a.bbox,
        bbox_3035=[minx, miny, maxx, maxy],
        crs=str(crs_ref),
        affine=affine_ref.to_gdal() if affine_ref else None,
        time_index=time_index,
        channels=channels,
        tensor_shape=tensor.shape,
        layout=a.layout,
        dtype=str(tensor.dtype),
        normalised=a.normalise,
        log1p=a.log1p,
        sentinel=a.sentinel,
        mean_active=float(mu),
        std_active=float(std),
        generated_utc=datetime.utcnow().isoformat(timespec="seconds")+"Z",
    )
    meta_path = a.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    LOG.info("Meta JSON → %s", meta_path)
    LOG.info("Done.")

if __name__ == "__main__":
    main()
