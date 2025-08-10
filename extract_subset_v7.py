#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#  extract_subset_v7.py – v7 데이터셋 서브셋터 (대륙/임의 ISO3)
#  • v7 산출 구조 100% 호환:
#      features.pt(X,S,M,mask_S), labels_all.npz(모든 Y_/Ymask_ 키 포함),
#      sample_index.parquet, dataset_meta.json, country_master.csv, splits.pt,
#      area_order.csv
#  • out-dir에 신규 area_order.csv( new/old/m49/iso3/name ) 저장
#  • v7 철학 반영: 서브셋 기준의 label_stats_yearly.csv/.parquet 재계산 저장
# -----------------------------------------------------------------------------
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path
from typing import List, Set, Tuple, Dict

import numpy as np
import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
import torch

LOG = logging.getLogger("extract_v7")
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

# ────────────────────────────────────────────────────────────────────────────
# 대륙 식별 (pycountry-convert 사용 가능 시 활용, 아니면 아프리카 상수셋)
try:
    import pycountry_convert as pcc
    def in_continent(iso3: str, continent_code: str) -> bool:
        try:
            iso2 = pcc.country_alpha3_to_country_alpha2(iso3)
            return pcc.country_alpha2_to_continent_code(iso2) == continent_code
        except Exception:
            return False
except Exception:
    AFR_ISO3 = {
        "DZA","AGO","BEN","BWA","BFA","BDI","CMR","CPV","CAF","TCD","COM","COD",
        "COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN",
        "GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ",
        "NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN",
        "TZA","TGO","TUN","UGA","ZMB","ZWE",
    }
    def in_continent(iso3: str, continent_code: str) -> bool:
        return (continent_code == "AF") and (iso3 in AFR_ISO3)

# ────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--in-dir",  required=True,  type=Path, help="v7 산출 경로(날짜 폴더)")
    p.add_argument("--out-dir", required=True,  type=Path, help="서브셋 저장 경로")

    # 필터 옵션 (셋 중 하나 이상)
    p.add_argument("--continent", type=str, default="AF",
                   help="대륙 코드(예: AF=Africa). ISO3 옵션이 없으면 이 값 사용")
    p.add_argument("--iso3-list", type=str, default="",
                   help="콤마구분 ISO3 목록 예: 'NGA,KEN,ETH'")
    p.add_argument("--iso3-csv",  type=Path,
                   help="한 줄에 하나씩 ISO3가 있는 파일(CSV/TXT, 헤더 불필요)")

    # 고급: area 순서를 외부에서 명시하고 싶을 때
    p.add_argument("--area-order-csv", type=Path,
                   help="(선택) 입력 데이터셋의 area 순서 CSV. 컬럼: m49[, iso3, name]. "
                        "미지정 시 in-dir/area_order.csv 사용, 없으면 휴리스틱.")

    return p.parse_args()

# ────────────────────────────────────────────────────────────────────────────
def load_pack(in_dir: Path):
    feat  = torch.load(in_dir / "features.pt", map_location="cpu", weights_only=False)
    meta  = json.loads((in_dir / "dataset_meta.json").read_text())
    labs  = np.load(in_dir / "labels_all.npz")
    df_si = pq.read_table(in_dir / "sample_index.parquet").to_pandas()
    cm_fn = in_dir / "country_master.csv"
    if not cm_fn.exists():
        raise SystemExit("country_master.csv 가 없습니다. build_dataset_v7.py 산출물이 맞는지 확인하세요.")
    cm    = pd.read_csv(cm_fn)
    splits= torch.load(in_dir / "splits.pt", map_location="cpu", weights_only=False)
    # 날짜 인덱스
    if "time_index" in meta:
        dates = pd.to_datetime(meta["time_index"], format="%Y%m", utc=True)
    else:
        T = int(meta.get("T", feat["S"].shape[1]))
        dates = pd.date_range("2000-01-01", periods=T, freq="MS", tz="UTC")
    return feat, meta, labs, df_si, cm, splits, dates

def decide_iso3_set(args, cm: pd.DataFrame) -> Set[str]:
    iso3_set: Set[str] = set()
    if args.iso3_csv:
        s = set(pd.read_csv(args.iso3_csv, header=None)[0].astype(str).str.upper().str.strip())
        iso3_set |= s
    if args.iso3_list:
        s = set([x.strip().upper() for x in args.iso3_list.split(",") if x.strip()])
        iso3_set |= s
    if not iso3_set:
        cont = args.continent.strip().upper()
        for iso3 in cm["iso3"].dropna().astype(str).str.upper().unique():
            if in_continent(iso3, cont):
                iso3_set.add(iso3)
        if cont == "AF" and len(iso3_set) == 0:
            LOG.warning("대륙 필터 결과가 비었습니다. ISO3 옵션을 제공하거나 continent 코드를 확인하세요.")
    return iso3_set

def infer_area_order(in_dir: Path, cm: pd.DataFrame, n_dataset: int, override_csv: Path|None) -> List[int]:
    """
    area(국가) 순서를 m49 코드 리스트로 반환.
    우선순위: override_csv → in-dir/area_order.csv → 휴리스틱
    """
    order_file = override_csv if override_csv else (in_dir / "area_order.csv")
    if order_file and order_file.exists():
        df = pd.read_csv(order_file)
        if "m49" not in df.columns:
            raise ValueError("area_order.csv에는 'm49' 컬럼이 필요합니다.")
        order = df["m49"].astype(int).tolist()
        if len(order) != n_dataset:
            LOG.warning("area_order.csv 길이(%d) ≠ n_countries(%d). 상위 n만 사용.",
                        len(order), n_dataset)
            order = order[:n_dataset]
        return order

    # 휴리스틱: m49 정렬 후 상위 n_dataset개 → v7 빌더의 sorted(common)과 일치
    LOG.warning("area_order.csv 없음 → 휴리스틱 사용(정렬 후 상위 n). "
                "정확한 매핑을 위해 area_order.csv 제공을 권장합니다.")
    m49_all = (cm.drop_duplicates("m49")["m49"].dropna().astype(int).tolist())
    m49_all = sorted(m49_all)
    if len(m49_all) < n_dataset:
        raise RuntimeError("country_master.csv의 m49 개수가 n_countries보다 적습니다.")
    return m49_all[:n_dataset]

def build_index_maps(area_m49_order: List[int]) -> Dict[int,int]:
    """ m49 → area_ord(0..k-1) 매핑 """
    return {m49: idx for idx, m49 in enumerate(area_m49_order)}

def subset_indices(cm: pd.DataFrame,
                   area_idx_map: Dict[int,int],
                   iso3_keep: Set[str]) -> Tuple[List[int], List[int], pd.DataFrame]:
    """
    반환: (keep_area_ord 리스트, keep_m49 리스트, cm_sub DataFrame)
    """
    cm_u = cm.drop_duplicates("m49").copy()
    cm_u["iso3"] = cm_u["iso3"].astype(str).str.upper()
    keep_area, keep_m49 = [], []
    rows=[]
    for _,r in cm_u.iterrows():
        if r["iso3"] in iso3_keep:
            old = area_idx_map.get(int(r["m49"]))
            if old is not None:
                keep_area.append(old); keep_m49.append(int(r["m49"]))
                rows.append(r)
    if not keep_area:
        raise SystemExit("❌ 선택된 ISO3가 데이터셋에 없습니다. 필터 조건을 확인하세요.")
    cm_sub = pd.DataFrame(rows).reset_index(drop=True)
    return sorted(keep_area), sorted(keep_m49), cm_sub

def save_area_order_csv(out_dir: Path,
                        keep_area: List[int],
                        cm_sub: pd.DataFrame,
                        area_idx_map: Dict[int,int]):
    inv = {v:k for k,v in area_idx_map.items()}  # area_ord -> m49
    recs=[]
    for new, old in enumerate(sorted(keep_area)):
        m49 = inv[old]
        row = cm_sub.loc[cm_sub["m49"]==m49].iloc[0]
        recs.append({
            "new_area_ord": new,
            "old_area_ord": old,
            "m49": int(m49),
            "iso3": str(row["iso3"]),
            "name": str(row.get("name", "")),
        })
    pd.DataFrame(recs).to_csv(out_dir / "area_order.csv", index=False)

def recompute_label_stats_yearly(labels_npz: dict,
                                 dates: pd.DatetimeIndex,
                                 out_dir: Path):
    """
    v7 QC 일치: base 라벨(Y_h, Ymask_h)만 사용(멀티-thr 제외)
    """
    # 키 수집
    y_keys = [k for k in labels_npz.files if k.startswith("Y_") and "_thr" not in k and "mask" not in k.lower()]
    h_list = sorted([int(k.split("_")[1]) for k in y_keys])
    rows=[]
    years = sorted(dates.year.unique())
    for h in h_list:
        Yk = f"Y_{h}"
        Mk = f"Ymask_{h}"
        if (Yk not in labels_npz) or (Mk not in labels_npz):
            continue
        Y  = labels_npz[Yk]         # (A', T)
        Mv = labels_npz[Mk]         # (A', T)
        for y in years:
            mask_y = (dates.year==y)
            valid  = int(Mv[:, mask_y].sum())
            poss   = int((Y[:, mask_y] * Mv[:, mask_y]).sum())
            pr = float(poss) / float(max(valid,1))
            rows.append({"horizon":h, "year":int(y),
                         "valid":valid, "positives":poss, "pos_rate":pr})
    df_stats = pd.DataFrame(rows).sort_values(["horizon","year"])
    # 저장
    pq.write_table(pa.Table.from_pandas(df_stats), out_dir / "label_stats_yearly.parquet")
    df_stats.to_csv(out_dir / "label_stats_yearly.csv", index=False)
    LOG.info("✓ Recomputed label_stats_yearly for subset.")

# ────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    feat, meta, labs, df_si, cm, splits, dates = load_pack(args.in_dir)
    A = int(meta["n_countries"])
    T = int(meta.get("T", feat["S"].shape[1]))
    LOG.info(f"Loaded: A={A}  T={T}  horizons={sorted(meta['horizons'])}")

    # sanity
    if df_si["area_ord"].max() >= A:
        raise AssertionError("sample_index.area_ord 가 n_countries 범위를 초과합니다.")

    # (1) 데이터셋 area 순서(m49 리스트) 확정
    area_m49_order = infer_area_order(args.in_dir, cm, A, args.area_order_csv)
    area_idx_map   = build_index_maps(area_m49_order)

    # (2) 필터 ISO3 집합
    iso3_keep = decide_iso3_set(args, cm)
    if not iso3_keep:
        LOG.error("ISO3 필터가 비었습니다. --iso3-list / --iso3-csv / --continent 를 확인하세요.")
        sys.exit(1)
    LOG.info(f"Filter size: {len(iso3_keep)} ISO3")

    # (3) 유지할 area/ m49 계산
    keep_area, keep_m49, cm_sub = subset_indices(cm, area_idx_map, iso3_keep)
    LOG.info(f"✓ Subset countries present in dataset: {len(keep_area)}")

    # (4) features 필터링 (X는 국가 차원 없음)
    feat_out = {
        "X":       feat["X"],                    # (N_win, L, C, H, W)
        "S":       feat["S"][keep_area],         # (A', T, D)
        "M":       feat["M"][keep_area],         # (A', T, Md)
        "mask_S":  feat["mask_S"][keep_area],    # (A', T, D)
    }
    torch.save(feat_out, args.out_dir / "features.pt")

    # (5) labels_all.npz 필터: 모든 키(Y_*, Ymask_* 포함) shape==(A,T)만 자르고, 나머지는 원본 유지
    lab_sub = {}
    for k in labs.files:
        v = labs[k]
        if (isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[0] == A):
            lab_sub[k] = v[keep_area]
        else:
            lab_sub[k] = v  # shape 불일치(스칼라/메타 등) → 원본 유지
    np.savez_compressed(args.out_dir / "labels_all.npz", **lab_sub)

    # (6) sample_index.parquet 갱신 (area 재인덱싱)
    old2new = {old: new for new, old in enumerate(sorted(keep_area))}
    tbl_sub = df_si[df_si.area_ord.isin(old2new)].copy()
    tbl_sub["area_ord"] = tbl_sub["area_ord"].map(old2new)
    pq.write_table(pa.Table.from_pandas(tbl_sub), args.out_dir / "sample_index.parquet")

    # (7) splits.pt 원본 그대로 복사 (국가와 무관)
    torch.save(splits, args.out_dir / "splits.pt")

    # (8) country_master / meta 저장
    cm_out = cm.drop_duplicates("m49")
    cm_out = cm_out[cm_out["m49"].isin(keep_m49)].copy()
    cm_out.to_csv(args.out_dir / "country_master.csv", index=False)

    meta_out = dict(meta)
    meta_out["n_countries"] = len(keep_area)
    meta_out["subset_note"] = "Subset generated by extract_subset_v7.py"
    meta_out["subset_source"] = str(args.in_dir.resolve())
    (args.out_dir / "dataset_meta.json").write_text(json.dumps(meta_out, indent=2))

    # (9) 신규 area_order.csv(재현성/추적성)
    save_area_order_csv(args.out_dir, keep_area, cm_sub, area_idx_map)

    # (10) v7 철학: 서브셋 기준 label_stats_yearly.* 재계산
    recompute_label_stats_yearly(np.load(args.out_dir / "labels_all.npz"), dates, args.out_dir)

    # (11) 요약
    LOG.info(f"Saved → {args.out_dir.resolve()}")
    LOG.info("Top-10 kept countries:")
    inv = {v:k for k,v in area_idx_map.items()}
    for old in sorted(keep_area)[:10]:
        m49 = inv[old]
        row = cm_out.loc[cm_out["m49"]==m49].iloc[0]
        LOG.info(f" · [{row['iso3']}] {row.get('name','')}  (old:{old} → new:{old2new[old]})")

if __name__ == "__main__":
    main()
