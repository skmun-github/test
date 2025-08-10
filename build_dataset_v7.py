#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  build_dataset_v7.py – EMODnet v4 + Static Indicators 파이프라인 (EWS 최적화)
#  • v5 대비 주요 보강
#    - IFPA 기본 기준연도: 2000:2018 (분포 이동 방지 실험 설계 반영)
#    - dataset_meta.json에 time_index(YYYYMM) 저장
#    - area_order.csv 저장 (old/new area_ord, m49, iso3, name)
#    - label_stats_yearly.csv 저장 (연도·horizon별 유효/양성/양성비)
#    - Ymask 정의: 미래 윈도우 전체 관측 가능한 시점만 1 (빈티지-세이프)
#    - parquet 저장 경로 정리 (pa.Table.from_pandas + pq.write_table)
#  • 산출물: features.pt / labels_all.npz / sample_index.parquet /
#            dataset_meta.json / country_master.csv / area_order.csv /
#            label_stats_yearly.csv
# =============================================================================

from __future__ import annotations
import argparse, json, logging, math, re, sys, time, unicodedata, warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

SENTINEL_STATIC = -999.0

# ────────────── 로깅 ────────────────────────────────────────────────────────
LOG = logging.getLogger("build_v7")
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)

# ────────────── CLI ────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 필수 입력
    p.add_argument("--npz",          required=True,  type=Path)
    p.add_argument("--meta-json",    required=True,  type=Path,
                   help="EMODnet 텐서 메타(json). time_index(YYYYMM) 포함 권장")
    p.add_argument("--price-csv",    required=True,  type=Path)
    p.add_argument("--crops-csv",    type=Path)
    p.add_argument("--gdp-csv",      type=Path)
    p.add_argument("--wdi-xlsx",     type=Path)
    p.add_argument("--raw-dir",      required=True,  type=Path)
    p.add_argument("--out-dir",      required=True,  type=Path)
    p.add_argument("--skip-static",  action="store_true",
                   help="정적 지표를 완전히 생략 (S/mask_S 빈 텐서)")

    # 윈도·지평
    p.add_argument("--window-len",   type=int, default=12)
    p.add_argument("--horizon",      nargs="+", type=int, default=[1,3])

    # 스케일·인코딩
    p.add_argument("--scaler", choices=["none","zscore","robust","minmax","pct","log"],
                   default="robust")
    p.add_argument("--month-enc", choices=["onehot","cyclical"], default="cyclical")

    # Surge 라벨(탐지 논리)
    p.add_argument("--surge-mode", choices=["rolling_sigma","percentile","pot","ifpa"],
                   default="ifpa")
    p.add_argument("--sigma-mult", type=float, default=1.3,
                   help="rolling_sigma: μ + c·σ 의 c 계수")
    p.add_argument("--roll-k",     type=int,   default=6,
                   help="rolling_sigma: 롤링 윈도 크기(개월)")
    p.add_argument("--min-obs",    type=int,   default=3,
                   help="rolling 통계 최소 관측치 (rolling_sigma)")
    p.add_argument("--percentile", type=float, default=95,
                   help="percentile 임계 (상위 p%)")
    p.add_argument("--pot-thr",    type=float, default=0.05,
                   help="POT exceedance 비율 (상위 q)")

    # IFPA 파라미터
    p.add_argument("--ifpa-gamma", type=float, default=0.4,
                   help="IFPA: 3M vs 12M 가중치 γ (0~1)")
    p.add_argument("--ifpa-thr",   type=float, default=1.0,
                   help="IFPA: z-score threshold (단일)")

    # IFPA 기준분포/빈티지
    p.add_argument("--baseline-mode", choices=["fixed","trailing"], default="fixed",
                   help="IFPA 기준분포: fixed(연도구간) 또는 trailing(W년)")
    p.add_argument("--baseline-years", type=str, default="2000:2018",
                   help="fixed 모드 기준 연도 범위, 예: 2000:2018")
    p.add_argument("--trailing-years", type=int, default=10,
                   help="trailing 모드 기준 창 길이(년)")
    p.add_argument("--vintage-interp", choices=["ffill","none"], default="ffill",
                   help="라벨용 가격 보간: ffill만 허용(양방향 금지)")

    # 라벨 타이밍/온셋
    p.add_argument("--label-timing", choices=["exact","window","onset"], default="window",
                   help="exact: t+h, window: 다음 h개월 내, onset: 온셋이 다음 h개월 내")
    p.add_argument("--event-window", type=int, default=None,
                   help="window 라벨에서 사용할 윈도 길이(기본: horizon과 동일)")
    p.add_argument("--refractory", type=int, default=2,
                   help="온셋 간 최소 간격(개월), onset 모드에서 사용")
    p.add_argument("--min-duration", type=int, default=1,
                   help="z>=thr 지속 최소 개월 수(에피소드 인정 길이)")

    # 다중 임계(선택)
    p.add_argument("--multi-thr", type=str, default="",
                   help="쉼표구분 임계 다중 출력(예: '1.5,2.0,2.5'); 빈 문자열이면 단일 thr")

    # WDI 지표 필터
    p.add_argument("--wdi-nan-cut", type=float, default=0.6)

    # CV split (v7은 기본 산출만, 학습용 스플릿은 train에서 연도 기반 구성 권장)
    p.add_argument("--cv-stride", type=int, default=0,
                   help=">0 → Rolling‑Origin stride (개월)")
    p.add_argument("--val-year", type=int, choices=[2022, 2023], default=2022,
                   help="Validation 으로 사용할 연도 (나머지 한 해는 test 로 자동 지정)")

    # Sparse
    p.add_argument("--sparse-pt", type=Path)
    p.add_argument("--use-sparse", action="store_true")

    # 기타
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")

    return p.parse_args()

# ────────────── 유틸 ────────────────────────────────────────────────────────
_MONTH_MAP = {m: i for i,m in enumerate(
    ["January","February","March","April","May","June","July","August",
     "September","October","November","December"], 1)}

def _slug(text) -> str:
    """NaN·None 안전 slugify"""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    txt = unicodedata.normalize("NFKD", str(text))
    txt = re.sub(r"[^A-Za-z0-9]+", " ", txt).strip().lower()
    return re.sub(r"\s+", " ", txt)

def month_onehot(dates:pd.DatetimeIndex)->torch.Tensor:
    eye=torch.eye(12,dtype=torch.float32)
    return eye[[d.month-1 for d in dates]]

def month_cyclical(dates:pd.DatetimeIndex)->torch.Tensor:
    rad=2*math.pi*(dates.month-1)/12
    return torch.tensor(np.c_[np.sin(rad),np.cos(rad)],dtype=torch.float32)

# ────────────── 국가 매핑 ───────────────────────────────────────────────────
def build_country_table(raw:Path)->pd.DataFrame:
    import pycountry
    frames=[]
    for fn in ["CPIs.csv","Crops_and_livestock_products.csv",
               "Value_of_Agricultural_Production.csv"]:
        df=pd.read_csv(raw/fn,usecols=["Area Code (M49)","Area"])
        df=df.rename(columns={"Area Code (M49)":"m49","Area":"name"})
        frames.append(df)
    fao=pd.concat(frames,ignore_index=True).drop_duplicates()
    fao["slug"]=fao["name"].map(_slug)

    wdi=pd.read_excel(raw/"WDI_wo_CPI_related.xlsx",
                      usecols=["Country Name","Country Code"])\
         .rename(columns={"Country Name":"name","Country Code":"iso3"})
    wdi["slug"]=wdi["name"].map(_slug)

    iso_rows=[{"iso3":c.alpha_3,"iso2":c.alpha_2,"slug":_slug(c.name)}
              for c in pycountry.countries]
    iso_df=pd.DataFrame(iso_rows)

    master=(fao.merge(iso_df[["slug","iso3","iso2"]],on="slug",how="left")
               .merge(wdi[["slug","iso3"]],on="slug",how="left",suffixes=("","_w")))
    master["iso3"]=master["iso3"].fillna(master.pop("iso3_w"))
    alias={"china taiwan province of":"taiwan","china hong kong sar":"hong kong",
           "china macao sar":"macau","sudan former":"sudan",
           "congo democratic republic of the":"democratic republic of the congo",
           "bolivia plurinational state of":"bolivia",
           "venezuela bolivarian republic of":"venezuela"}
    master["slug"]=master["slug"].replace(alias)
    return master.drop_duplicates("slug").dropna(subset=["iso3"])

# ────────────── tidy 함수 ───────────────────────────────────────────────────
def tidy_cpi(p:Path,m49:set[int])->pd.DataFrame:
    use = ["Area Code (M49)","Year","Months","Item","Value"]
    df=pd.read_csv(p,usecols=use)
    df=df[df["Year"].between(2000,2024)]
    df=df[df["Item"].isin(['Consumer Prices, Food Indices (2015 = 100)',
                           'Consumer Prices, General Indices (2015 = 100)'])]
    df["month"]=df["Months"].map(_MONTH_MAP)
    df["date"]=pd.to_datetime(dict(year=df["Year"],month=df["month"],day=1),utc=True)
    df=df.rename(columns={"Area Code (M49)":"m49","Item":"item","Value":"value"})
    df=df[df["m49"].isin(m49)]
    df["value"]=pd.to_numeric(df["value"],errors="coerce").astype("float32")
    df.loc[:, "value"]=df["value"].replace([np.inf, -np.inf], np.nan)
    return df.pivot_table(index=["m49","date"],columns="item",values="value").sort_index()

def tidy_crops(p:Path,m49:set[int])->pd.DataFrame:
    df=pd.read_csv(p)
    df=df[df["Year"].between(2017,2023) & (df["Flag"]!="M")]
    df=df.rename(columns={"Area Code (M49)":"m49"})
    df=df[df["m49"].isin(m49)]
    df["var"]=df["Element"].str[0]+"_"+df["Item"].str.split().str[0]
    df["value"]=pd.to_numeric(df["Value"],errors="coerce").astype("float32")
    df.loc[:, "value"]=df["value"].replace([np.inf, -np.inf], np.nan)
    return df[["m49","Year","var","value"]]

def tidy_gpv(p:Path,m49:set[int])->pd.DataFrame:
    df=pd.read_csv(p)
    df=df[df["Year"].between(2017,2023)]
    df=df.rename(columns={"Area Code (M49)":"m49"})
    df=df[df["m49"].isin(m49)]
    df["var"]="GPV_"+df["Item"].str.split().str[0]
    df["value"]=pd.to_numeric(df["Value"],errors="coerce").astype("float32")
    df.loc[:, "value"]=df["value"].replace([np.inf, -np.inf], np.nan)
    return df[["m49","Year","var","value"]]

def tidy_wdi(xlsx:Path,m49_from_iso:dict[str,int],nan_cut:float)->pd.DataFrame:
    raw = pd.read_excel(xlsx, na_values="..")
    year_cols = [c for c in raw.columns if re.match(r"^\d{4}", str(c))]
    raw = raw.rename(columns={c: int(str(c)[:4]) for c in year_cols})
    keep = ["Country Code","Series Code"] + [int(str(c)[:4]) for c in year_cols]
    df = raw[keep].melt(id_vars=["Country Code","Series Code"],
                        var_name="Year", value_name="value")
    df=df.rename(columns={"Country Code":"iso3","Series Code":"var"})
    df["Year"]=df["Year"].astype(int)
    df["value"]=pd.to_numeric(df["value"], errors="coerce").astype("float32")
    df.loc[:, "value"]=df["value"].replace([np.inf, -np.inf], np.nan)
    df["m49"]=df["iso3"].map(m49_from_iso)
    df = df[~df["m49"].isna()]
    ok_vars = (df.groupby("var")["value"]
                 .apply(lambda v: v.isna().mean() <= nan_cut)
                 .pipe(lambda s: s[s].index))
    return df[df["var"].isin(ok_vars)][["m49","Year","var","value"]]

# ────────────── 연→월 복제(빈티지-세이프) ──────────────────────────────────
def replicate_annual(df:pd.DataFrame,dates:pd.DatetimeIndex,m49:list[int])->pd.DataFrame:
    wide=(df.pivot_table(index=["m49","Year"],columns="var",values="value")
            .sort_index())
    rows=[]
    for d in dates:
        midx=pd.MultiIndex.from_product([m49,[d.year-1]],names=["m49","Year"])
        rows.append(wide.reindex(midx).droplevel("Year").assign(date=d).reset_index())
    return (pd.concat(rows,ignore_index=True)
              .set_index(["m49","date"])
              .sort_index())

# ────────────── 스케일러 ───────────────────────────────────────────────────
class Scaler:
    def __init__(self,mode:str="robust",eps=1e-6):
        self.mode=mode; self.eps=eps; self.stats:dict[str,pd.Series]={}
    def fit(self,x:pd.DataFrame):
        if self.mode=="none": return
        if self.mode=="zscore":
            self.stats["mean"]=x.mean(); self.stats["std"]=x.std().replace(0,1)
        elif self.mode=="robust":
            self.stats["med"]=x.median()
            self.stats["iqr"]=x.quantile(.75)-x.quantile(.25)
            self.stats["iqr"].replace(0,1,inplace=True)
        elif self.mode=="minmax":
            self.stats["min"]=x.min(); self.stats["max"]=x.max()
    def transform(self,x:pd.DataFrame)->pd.DataFrame:
        if self.mode=="none": return x
        if self.mode=="zscore":
            return (x-self.stats["mean"])/self.stats["std"]
        if self.mode=="robust":
            return (x-self.stats["med"])/(self.stats["iqr"]+self.eps)
        if self.mode=="minmax":
            return (x-self.stats["min"])/(self.stats["max"]-self.stats["min"]+self.eps)
        if self.mode=="log":
            return np.log1p(x)
        if self.mode=="pct":
            return x.groupby(level=0).pct_change().fillna(0)
        raise ValueError

# ────────────── IFPA z 계산 ────────────────────────────────────────────────
def ifpa_z(price: pd.Series, gamma: float,
           baseline_mode: str, baseline_years: str, trailing_years: int) -> pd.Series:
    cq = np.log(price / price.shift(3))
    ca = np.log(price / price.shift(12))
    if baseline_mode == "fixed":
        a,b = map(int, baseline_years.split(":"))
        base = price[(price.index.year>=a) & (price.index.year<=b)]
        b_cq = np.log(base / base.shift(3))
        b_ca = np.log(base / base.shift(12))
        mu_cq = b_cq.groupby(b_cq.index.month).mean()
        sd_cq = b_cq.groupby(b_cq.index.month).std(ddof=0)
        mu_ca = b_ca.groupby(b_ca.index.month).mean()
        sd_ca = b_ca.groupby(b_ca.index.month).std(ddof=0)
        idx = price.index.month
        mu_cq = mu_cq.reindex(idx).to_numpy()
        sd_cq = sd_cq.reindex(idx).to_numpy()
        mu_ca = mu_ca.reindex(idx).to_numpy()
        sd_ca = sd_ca.reindex(idx).to_numpy()
        # ★ 안정화: mu NaN → 0, sd NaN/0 → 1e-6
        mu_cq[~np.isfinite(mu_cq)] = 0.0
        mu_ca[~np.isfinite(mu_ca)] = 0.0
        sd_cq[~np.isfinite(sd_cq) | (sd_cq == 0)] = 1e-6
        sd_ca[~np.isfinite(sd_ca) | (sd_ca == 0)] = 1e-6
        z = gamma*((cq.to_numpy()-mu_cq)/sd_cq) + (1-gamma)*((ca.to_numpy()-mu_ca)/sd_ca)
        z = pd.Series(z, index=price.index)
    else:
        W = trailing_years*12
        mu_cq = cq.rolling(W, min_periods=max(6, W//6)).mean()
        sd_cq = cq.rolling(W, min_periods=max(6, W//6)).std(ddof=0).replace(0,np.nan)
        mu_ca = ca.rolling(W, min_periods=max(12, W//6)).mean()
        sd_ca = ca.rolling(W, min_periods=max(12, W//6)).std(ddof=0).replace(0,np.nan)
        z = gamma*((cq-mu_cq)/sd_cq) + (1-gamma)*((ca-mu_ca)/sd_ca)
    return z.astype("float32")

# ────────────── 라벨 유틸 ──────────────────────────────────────────────────
def enforce_min_duration(flag: np.ndarray, min_len: int) -> np.ndarray:
    if min_len <= 1: return flag.astype(np.uint8)
    out = np.zeros_like(flag, dtype=np.uint8)
    i=0; n=len(flag)
    while i<n:
        if flag[i]:
            j=i
            while j<n and flag[j]: j+=1
            if (j-i) >= min_len: out[i:j] = 1
            i=j
        else:
            i+=1
    return out

def onset_only(flag: np.ndarray, refractory: int) -> np.ndarray:
    starts = (flag.astype(bool) & ~np.r_[False, flag[:-1].astype(bool)]).astype(np.uint8)
    if refractory <= 0: return starts
    out = np.zeros_like(starts, dtype=np.uint8)
    last = -10**9
    for t in np.where(starts==1)[0]:
        if t - last > refractory:
            out[t] = 1
            last = t
    return out

def window_labels(flag_src: np.ndarray, valid_src: np.ndarray,
                  h: int, mode: str, event_window: int|None) -> tuple[np.ndarray, np.ndarray]:
    """
    Y[t]=1 정의:
      - exact : Y[t]=flag[t+h]
      - window: 미래 [t+1, t+ew] 구간에 flag가 하나라도 있으면 1
      - onset : flag_src를 온셋만 남긴 뒤 window 로직 적용
    마스크 M[t]=1 정의:
      - exact : t+h 시점의 valid_src가 True
      - window/onset : 미래 윈도우 전체가 관측(valid) 가능할 때만 1 (빈티지-세이프)
    """
    T = len(flag_src)
    ew = event_window or h

    if mode == "exact":
        y = np.r_[flag_src[h:], np.zeros(h, dtype=np.uint8)]
        m = np.r_[valid_src[h:], np.zeros(h, dtype=bool)]
        return y, m.astype(np.uint8)

    y = np.zeros(T, dtype=np.uint8)
    m = np.zeros(T, dtype=bool)
    for t in range(T):
        end = min(T, t+ew+1)

        win_len = end - (t+1)
        if win_len < ew:                      # ★ 부분 윈도우 불허
            y[t] = 0
            m[t] = False
            continue
        win_flags = flag_src[t+1:end]
        win_valid = valid_src[t+1:end]
        is_full = (win_valid.size == ew) and win_valid.all()
        m[t] = is_full
        y[t] = 1 if (is_full and (win_flags.max() > 0)) else 0

    return y, m.astype(np.uint8)

# ────────────── Surge 라벨 생성 ────────────────────────────────────────────
def make_surge(df_cpi: pd.DataFrame, dates: pd.DatetimeIndex, h_list: List[int],
               mode: str, k: int, c: float, p: int, q: float, min_obs: int,
               **kwargs
               ) -> Tuple[Dict[int,np.ndarray], Dict[int,np.ndarray], List[int]]:
    m49=sorted(df_cpi.index.get_level_values(0).unique())
    T=len(dates)
    surge={h:np.zeros((len(m49),T),np.uint8) for h in h_list}
    ymask={h:np.zeros((len(m49),T),np.uint8) for h in h_list}

    for r,country in enumerate(tqdm(m49,"surge",leave=False)):
        s0=(df_cpi.xs(country,level=0)
              ["Consumer Prices, Food Indices (2015 = 100)"]
              .reindex(dates))
        price = s0.ffill() if kwargs.get("vintage_interp","ffill")=="ffill" else s0.copy()

        # z-score / flag 원천
        if mode=="ifpa":
            z = ifpa_z(price, kwargs.get("gamma", 0.4),
                       kwargs.get("baseline_mode","fixed"),
                       kwargs.get("baseline_years","2000:2018"),
                       kwargs.get("trailing_years",10))
            thr = kwargs.get("thr", 1.0)
            flag_src = (z.to_numpy() >= thr).astype(np.uint8)
            valid = np.isfinite(z.to_numpy())

        elif mode=="rolling_sigma":
            mom   = price.pct_change()
            mu    = mom.rolling(k, min_obs).mean().shift(1)
            sigma = mom.rolling(k, min_obs).std(ddof=0).shift(1).replace(0,np.nan)
            z     = ((mom - mu)/sigma).astype("float32")
            flag_src = (z.to_numpy() > c).astype(np.uint8)
            valid = np.isfinite(z.to_numpy())

        elif mode=="percentile":
            mom = price.pct_change()
            thr_val = np.nanpercentile(mom, p)
            z = (mom - thr_val).astype("float32")
            flag_src = (z.to_numpy() > 0).astype(np.uint8)
            valid = np.isfinite(z.to_numpy())

        elif mode=="pot":
            mom = price.pct_change()
            thr_val = mom.quantile(1-q)
            z = (mom - thr_val).astype("float32")
            flag_src = (z.to_numpy() > 0).astype(np.uint8)
            valid = np.isfinite(z.to_numpy())

        else:
            raise ValueError("Unknown surge mode")

        # 최소 지속/온셋 처리
        flag_src = enforce_min_duration(flag_src, kwargs.get("min_duration",1))
        label_base = onset_only(flag_src, kwargs.get("refractory",2)) \
                     if kwargs.get("label_timing","window")=="onset" else flag_src

        # horizon별 라벨/마스크
        for h in h_list:
            y, m = window_labels(label_base, valid, h,
                                 kwargs.get("label_timing","window"),
                                 kwargs.get("event_window", None))
            surge[h][r] = y
            ymask[h][r] = m

    return surge, ymask, m49

# ────────────── 동적 윈도우 ────────────────────────────────────────────────
def build_windows_dense(X:np.ndarray, L:int, h_min:int):
    T=X.shape[0]; ends=np.arange(L-1+h_min, T, dtype=np.int16)
    idx=ends[:,None]-h_min-np.arange(L)[None,:]+1
    return torch.tensor(X[idx],dtype=torch.float16), ends

# ────────────── CV Split (참고: train에서 연도 스플릿 권장) ────────────────
def split_masks(dates: pd.DatetimeIndex, stride: int, val_year: int = 2022) -> Dict[str, np.ndarray]:
    n = len(dates)
    m_val   = np.zeros(n, bool)
    m_test  = np.zeros(n, bool)
    if stride <= 0:
        test_year = 2023 if val_year == 2022 else 2022
        assert val_year != test_year
        m_val [dates.year == val_year ] = True
        m_test[dates.year == test_year] = True
    else:
        stride = max(1, stride)
        anchor = 0
        while anchor < n:
            val_start, val_end = anchor, min(anchor + stride, n)
            m_val[val_start:val_end] = True
            test_start, test_end = val_end, min(val_end + stride, n)
            m_test[test_start:test_end] = True
            anchor = test_end + stride
    m_train = ~(m_val | m_test)
    return {"train": m_train, "val": m_val, "test": m_test}

# ────────────── Main ───────────────────────────────────────────────────────
def main():
    args=parse_args(); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.verbose: LOG.setLevel(logging.DEBUG)

    t0=time.time(); out=(args.out_dir/time.strftime("%Y%m%d"))
    out.mkdir(parents=True,exist_ok=True)

    # 국가 매핑
    ct=build_country_table(args.raw_dir)
    ct.to_csv(out/"country_master.csv",index=False)
    m49_set=set(ct["m49"]); m49_from_iso=dict(zip(ct["iso3"],ct["m49"]))

    # EMODnet 텐서
    npz=np.load(args.npz,allow_pickle=False)["X"].astype(np.float32)    
    meta_in=json.loads(args.meta_json.read_text())
    if "time_index" in meta_in:
        dates=pd.to_datetime(meta_in["time_index"],format="%Y%m",utc=True)
    else:
        # 최후수단: 메타에 time_index가 없을 때 2000-01부터 길이에 맞게 생성
        LOG.warning("meta-json에 'time_index'가 없어 2000-01부터 synthetic index를 생성합니다.")
        T = npz.shape[0]
        dates = pd.date_range("2000-01-01", periods=T, freq="MS", tz="UTC")

    # ★ 안전장치: EMODnet T와 time_index 길이 일치 확인
    if len(dates) != npz.shape[0]:
        raise ValueError(
            f"time_index length ({len(dates)}) != EMODnet T ({npz.shape[0]}). "
            "meta-json 또는 npz를 확인하세요."
        )

    # Sparse 입력 선택
    if args.use_sparse and args.sparse_pt and args.sparse_pt.exists():
        sp=torch.load(args.sparse_pt)
        X=torch.sparse_coo_tensor(sp["coords"],sp["values"].float(),tuple(sp["shape"]))
        npz=X.to_dense().numpy().astype(np.float32)
        LOG.info("Sparse COO ➜ dense loaded (shape %s)",npz.shape)

    # 동적 윈도우
    X_win,t_end=build_windows_dense(npz,args.window_len,min(args.horizon))
    LOG.info("X_win %s  windows=%d",X_win.shape,len(t_end)); del npz

    # CPI / 정적 지표
    df_cpi  = tidy_cpi(args.price_csv ,m49_set)
    if args.skip_static:
        df_crop = df_gpv = df_wdi = pd.DataFrame()
    else:
        df_crop = tidy_crops(args.crops_csv ,m49_set) if args.crops_csv else pd.DataFrame()
        df_gpv  = tidy_gpv (args.gdp_csv   ,m49_set) if args.gdp_csv  else pd.DataFrame()
        df_wdi  = tidy_wdi (args.wdi_xlsx  ,m49_from_iso,args.wdi_nan_cut) if args.wdi_xlsx else pd.DataFrame()

    # 교집합 국가
    common=set(df_cpi.index.get_level_values(0))
    for df in [df_crop,df_gpv,df_wdi]:
        if not df.empty: common &= set(df["m49"].unique())
    m49=sorted(common); LOG.info("Countries=%d",len(m49))

    # Month representation
    M = (month_cyclical(dates) if args.month_enc == "cyclical" else month_onehot(dates))
    M = M.unsqueeze(0).repeat(len(m49), 1, 1).half()

    # Static concat
    if args.skip_static or (df_crop.empty and df_gpv.empty and df_wdi.empty):
        S = torch.empty((len(m49), len(dates), 0), dtype=torch.float16)
        mask_S = torch.empty((len(m49), len(dates), 0), dtype=torch.uint8)
    else:
        frames = []
        if not df_crop.empty: frames.append(replicate_annual(df_crop, dates, m49))
        if not df_gpv .empty: frames.append(replicate_annual(df_gpv , dates, m49))
        if not df_wdi .empty: frames.append(replicate_annual(df_wdi , dates, m49))
        st_df = (pd.concat(frames, axis=1).sort_index()) if frames else \
                pd.DataFrame(index=pd.MultiIndex.from_product([m49, dates], names=["m49","date"]))
        mask_df = st_df.isna().astype("uint8")
        mask_np = mask_df.to_numpy().reshape(len(m49), len(dates), -1).astype("uint8")
        scaler = Scaler(args.scaler); scaler.fit(st_df)
        st_np = scaler.transform(st_df).to_numpy(np.float32)
        st_np[~np.isfinite(st_np)] = np.nan
        nan_mask = np.isnan(st_np)
        mask_np |= nan_mask.reshape(mask_np.shape)
        st_np[nan_mask]  = SENTINEL_STATIC
        st_np[~nan_mask] = np.sign(st_np[~nan_mask]) * np.log1p(np.abs(st_np[~nan_mask]))
        S = torch.tensor(st_np.reshape(len(m49), len(dates), -1)).half()
        mask_S = torch.tensor(mask_np, dtype=torch.uint8)

    # ── 라벨/마스크 생성 ─────────────────────────────────────────────
    label_dict: Dict[str, np.ndarray] = {}
    ymask_dict: Dict[str, np.ndarray] = {}

    surge, ymask, _ = make_surge(
        df_cpi.loc[m49], dates, args.horizon,
        args.surge_mode, args.roll_k, args.sigma_mult,
        args.percentile, args.pot_thr, args.min_obs,
        gamma=args.ifpa_gamma, thr=args.ifpa_thr,
        label_timing=args.label_timing, event_window=args.event_window,
        refractory=args.refractory, min_duration=args.min_duration,
        baseline_mode=args.baseline_mode, baseline_years=args.baseline_years,
        trailing_years=args.trailing_years, vintage_interp=args.vintage_interp
    )
    for h in sorted(args.horizon):
        label_dict[f"Y_{h}"]     = surge[h]
        ymask_dict[f"Ymask_{h}"] = ymask[h]

    if args.surge_mode == "ifpa" and args.multi_thr.strip():
        for thr_str in args.multi_thr.split(","):
            thr_val = float(thr_str.strip())
            surge_m, ymask_m, _ = make_surge(
                df_cpi.loc[m49], dates, args.horizon,
                args.surge_mode, args.roll_k, args.sigma_mult,
                args.percentile, args.pot_thr, args.min_obs,
                gamma=args.ifpa_gamma, thr=thr_val,
                label_timing=args.label_timing, event_window=args.event_window,
                refractory=args.refractory, min_duration=args.min_duration,
                baseline_mode=args.baseline_mode, baseline_years=args.baseline_years,
                trailing_years=args.trailing_years, vintage_interp=args.vintage_interp
            )
            for h in sorted(args.horizon):
                label_dict[f"Y_{h}_thr{thr_val:g}"]     = surge_m[h]
                ymask_dict[f"Ymask_{h}_thr{thr_val:g}"] = ymask_m[h]

    # ── split (참고용; 학습 시엔 연도 기반 스플릿 권장) ──────────────
    splits=split_masks(dates,args.cv_stride, args.val_year)
    torch.save({k:v.astype(bool) for k,v in splits.items()}, out/"splits.pt")

    # ── 저장: features ───────────────────────────────────────────────
    torch.save({"X":X_win.half(), "S":S, "M":M, "mask_S":mask_S}, out/"features.pt")

    # ── 저장: labels/masks ───────────────────────────────────────────
    #np.savez_compressed(out/"labels_all.npz", **label_dict, **ymask_dict)

    # ── 저장: labels/masks (dtype 일관화) ───────────────────────────
    label_dict = {k: v.astype(np.uint8, copy=False) for k, v in label_dict.items()}
    ymask_dict = {k: v.astype(np.uint8, copy=False) for k, v in ymask_dict.items()}
    np.savez_compressed(out/"labels_all.npz", **label_dict, **ymask_dict)

    # ── sample_index (학습 시 (win_idx, area_ord, horizon, t_end) 매핑) ───
    tbl=[(w,a,h_idx,int(t)) for w,t in enumerate(t_end)
         for a in range(len(m49))
         for h_idx,_ in enumerate(sorted(args.horizon))]
    df_idx = pd.DataFrame(tbl, columns=["win_idx","area_ord","h_idx","t_end"])
    pq.write_table(pa.Table.from_pandas(df_idx), out/"sample_index.parquet")

    # ── area_order.csv (재현성/서브셋 편의) ───────────────────────────
    #   new_area_ord == old_area_ord (빌더 최초 생성이므로 동일)
    cm = ct.drop_duplicates("m49")
    recs=[]
    for i,m in enumerate(m49):
        row = cm.loc[cm["m49"]==m].iloc[0]
        recs.append({"new_area_ord": i, "old_area_ord": i, "m49": int(m),
                     "iso3": str(row["iso3"]), "name": str(row.get("name",""))})
    pd.DataFrame(recs).to_csv(out/"area_order.csv", index=False)

    # ── label_stats_yearly.csv (진단/실험 설계에 도움) ───────────────
    stats=[]
    years = pd.Index(dates.year.unique())
    for key in label_dict.keys():
        if not key.startswith("Y_") or "_thr" in key: continue
        h = int(key.split("_")[1])
        Y = label_dict[key]; Mv = ymask_dict[f"Ymask_{h}"]
        for y in years:
            mask_y = (dates.year==y)
            valid  = Mv[:, mask_y].sum()
            poss   = (Y[:, mask_y] * Mv[:, mask_y]).sum()
            pr = float(poss) / float(max(valid,1))
            stats.append({"horizon":h, "year":int(y),
                          "valid":int(valid), "positives":int(poss),
                          "pos_rate":pr})
    df_stats = pd.DataFrame(stats).sort_values(["horizon","year"])
    pq.write_table(pa.Table.from_pandas(df_stats), out/"label_stats_yearly.parquet")
    df_stats.to_csv(out/"label_stats_yearly.csv", index=False)

    # ── 메타 저장 (train 스크립트 호환 키 포함) ──────────────────────
    meta_out=dict(
        T=len(dates), window_len=args.window_len, horizons=sorted(args.horizon),
        surge_mode=args.surge_mode, scaler=args.scaler, month_enc=args.month_enc,
        cv_stride=args.cv_stride, val_year=args.val_year,
        npz_shape=meta_in.get("tensor_shape", None),
        X_win=tuple(X_win.shape), static_dim=S.shape[-1]+M.shape[-1],
        n_windows=len(t_end), n_countries=len(m49), sentinel_static=SENTINEL_STATIC,
        label_timing=args.label_timing, event_window=args.event_window,
        refractory=args.refractory, min_duration=args.min_duration,
        ifpa_gamma=args.ifpa_gamma, ifpa_thr=args.ifpa_thr,
        baseline_mode=args.baseline_mode, baseline_years=args.baseline_years,
        trailing_years=args.trailing_years, vintage_interp=args.vintage_interp,
        multi_thr=args.multi_thr,
        time_index=[d.strftime("%Y%m") for d in dates],   # ★ train에서 필요
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
    )
    json.dump(meta_out, open(out/"dataset_meta.json","w"), indent=2)

    LOG.info("✅ Finished → %s (%.1fs)", out, time.time()-t0)

# ───────────────────────────────────────────────────────────────────────────
if __name__=="__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        main()
