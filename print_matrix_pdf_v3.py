#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
print_matrix_pdf_v3.py

목적
----
dist_post_pca_euclidean_long.dta (국가-연도 × 국가-연도, post-PCA 2D 유클리드 거리, 하삼각 long 형식)에서
(i,j) 값들을 "아주 빽빽한 PDF"로 출력하고, 동시에 결측·통계·일관성 점검을 매우 로버스트하게 수행.

핵심 특징
--------
- **DTA만 사용** (추가 파일 불필요)
- 기본 폰트 아주 작게(4.2pt), 기본 5열 → 한 페이지에 최대치로 많이 표시
- 기본 모드: **stride=2** (결정적 모듈러 샘플링, 전체의 약 1/2)  ← 사용자 요청 반영
- 현실적 안전장치:
  * 예상 페이지 수 계산 및 경고
  * --max-pages 로 하드캡 (기본 200페이지). 해제와 강제 전체출력도 가능
- 통계/결측 체크(스트리밍):
  * 총 행수/라벨수/기대 하삼각 행수(n(n+1)/2) 일치 여부
  * NaN/±Inf/음수/비대각 0(또는 극소) 개수
  * 대각선(distance≈0) 검증
  * Welford 알고리즘으로 평균/분산(비대각)
  * Reservoir sampling(기본 300k)으로 분위수 근사(비대각 전체 / 동일연도 / 교차연도)
  * 동일연도/교차연도 건수 및 분포 비교

사용 예시
--------
# 1) 기본: stride=2, 5열, 4.2pt, 최대 200페이지까지 출력 + 상세 통계 파일 저장
python print_matrix_pdf_v3.py \
  --dta ./vdem_outputs_decade/dist_post_pca_euclidean_long.dta \
  --out-pdf ./vdem_outputs_decade/printout_stride2.pdf

# 2) 대각만 출력(라벨수 만큼), 라벨까지 함께 표기
python print_matrix_pdf_v3.py \
  --dta ./vdem_outputs_decade/dist_post_pca_euclidean_long.dta \
  --out-pdf ./vdem_outputs_decade/printout_diag.pdf \
  --mode diagonal --include-labels yes --max-pages 50

# 3) 상단 1000×1000 창(window)만 출력 (지정 범위)
python print_matrix_pdf_v3.py \
  --dta ./vdem_outputs_decade/dist_post_pca_euclidean_long.dta \
  --out-pdf ./vdem_outputs_decade/printout_window_0_999.pdf \
  --mode window --i-max 999 --j-max 999 --max-pages 200

# 4) 전량 출력(매우 큼) - 강제. 위험: 수만~수십만 페이지 예상
python print_matrix_pdf_v3.py \
  --dta ./vdem_outputs_decade/dist_post_pca_euclidean_long.dta \
  --out-pdf ./vdem_outputs_decade/printout_ALL.pdf \
  --mode all --force-full yes --max-pages 0

필수 패키지
----------
pandas, numpy, matplotlib

권장
----
pandas >= 1.2 (Stata chunksize+columns 동시 사용 안정)
"""

from __future__ import annotations
import argparse
import gzip
import math
import os
import sys
from typing import Iterator, List, Optional, Dict, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ----------------------------- 유틸 -----------------------------

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def _iter_rows_stata(
    dta_path: str,
    columns: List[str],
    chunksize: int = 500_000,
):
    """
    .dta를 청크 단위로 읽는 제너레이터.
    pandas 버전에 따라 columns+chunksize 동시 미지원일 수 있음 → 대응.
    """
    try:
        it = pd.read_stata(
            dta_path,
            columns=columns,
            chunksize=chunksize,
            convert_categoricals=False,
        )
        for chunk in it:
            yield chunk
        return
    except TypeError:
        # columns 인자 미지원 → chunksize만 사용 후 슬라이스
        try:
            it = pd.read_stata(
                dta_path,
                chunksize=chunksize,
                convert_categoricals=False,
            )
            for chunk in it:
                if columns:
                    chunk = chunk[columns]
                yield chunk
            return
        except Exception as e:
            raise RuntimeError(
                "pandas 버전이 오래되어 .dta 청크 스트리밍이 불안정합니다. "
                "pandas>=1.2 업그레이드 권장. 원인: " + repr(e)
            )


def _count_labels_via_diag(dta_path: str) -> int:
    """
    라벨 수 = (i_idx==j_idx) 행 수. 청크 스트리밍으로 카운트.
    """
    cnt = 0
    for chunk in _iter_rows_stata(dta_path, columns=["i_idx", "j_idx"], chunksize=1_000_000):
        cnt += int((chunk["i_idx"].to_numpy(dtype=np.int64) == chunk["j_idx"].to_numpy(dtype=np.int64)).sum())
    return cnt


# ----------------------------- 레이아웃 계산 -----------------------------

def compute_layout(
    page_size: str = "letter",
    top_in: float = 0.35, bottom_in: float = 0.35,
    left_in: float = 0.35, right_in: float = 0.35,
    ncols: int = 5, fontsize_pt: float = 4.2, line_spacing: float = 1.0
) -> Dict[str, float]:
    """
    매우 작은 폰트/여백/열 수를 입력받아 페이지당 줄 수를 계산.
    """
    if page_size.lower() == "letter":
        w_in, h_in = 8.5, 11.0
    elif page_size.lower() == "a4":
        w_in, h_in = 8.27, 11.69
    else:
        raise ValueError("page_size must be 'letter' or 'a4'")
    avail_h_points = (h_in - top_in - bottom_in) * 72.0
    line_height_pt = max(3.8, fontsize_pt * line_spacing)
    lines_per_col = max(1, int(math.floor(avail_h_points / line_height_pt)))
    return {
        "page_w_in": w_in, "page_h_in": h_in,
        "top_in": top_in, "bottom_in": bottom_in,
        "left_in": left_in, "right_in": right_in,
        "ncols": ncols, "fontsize_pt": fontsize_pt,
        "line_spacing": line_spacing,
        "lines_per_col": lines_per_col,
        "lines_per_page": lines_per_col * ncols
    }


# ----------------------------- 라인 생성 -----------------------------

def lines_generator(
    dta_path: str,
    mode: str,
    precision: int = 4,
    index_base: int = 1,
    stride: int = 2,
    i_min: Optional[int] = None, i_max: Optional[int] = None,
    j_min: Optional[int] = None, j_max: Optional[int] = None,
    include_labels: bool = False,
) -> Iterator[str]:
    """
    DTA를 스트리밍하며 조건에 맞는 줄을 "(i,j) value"로 생성.
    - index_base=1 → (i_idx+1, j_idx+1) 표기
    - include_labels=True → " [LABELi vs LABELj]" 병기
    - stride 샘플링은 결정적으로 (i*1,000,003 + j) % stride == 0 규칙을 사용
    """
    cols = ["i_idx","j_idx","distance"]
    if include_labels:
        cols += ["i_label","j_label"]
    for chunk in _iter_rows_stata(dta_path, columns=cols, chunksize=500_000):
        ii = chunk["i_idx"].to_numpy(dtype=np.int64)
        jj = chunk["j_idx"].to_numpy(dtype=np.int64)
        dd = chunk["distance"].to_numpy(dtype=float)

        # 기본 마스크
        mask = np.ones(len(chunk), dtype=bool)
        if mode == "diagonal":
            mask &= (ii == jj)
        elif mode == "stride":
            if stride <= 1:
                pass
            else:
                mask &= (((ii * 1_000_003 + jj) % int(stride)) == 0)
        elif mode == "window":
            if i_min is not None: mask &= (ii >= i_min)
            if i_max is not None: mask &= (ii <= i_max)
            if j_min is not None: mask &= (jj >= j_min)
            if j_max is not None: mask &= (jj <= j_max)
        elif mode == "all":
            pass
        else:
            raise ValueError("mode must be one of: diagonal, stride, window, all")

        if not np.any(mask):
            continue

        ii_s = ii[mask] + (index_base - 0)
        jj_s = jj[mask] + (index_base - 0)
        dd_s = dd[mask]
        if include_labels:
            li = chunk.loc[mask, "i_label"].astype(str).to_numpy()
            lj = chunk.loc[mask, "j_label"].astype(str).to_numpy()
            for a, b, v, si, sj in zip(ii_s, jj_s, dd_s, li, lj):
                yield f"({a},{b}) {v:.{precision}g}  [{si} vs {sj}]"
        else:
            for a, b, v in zip(ii_s, jj_s, dd_s):
                yield f"({a},{b}) {v:.{precision}g}"


# ----------------------------- PDF 작성 -----------------------------

def write_pdf(
    out_pdf: str,
    lines: Iterator[str],
    layout: dict,
    title: Optional[str] = None,
):
    """
    lines를 받아 페이지당 columns×lines_per_col로 배치해 PDF 생성.
    """
    w_in = layout["page_w_in"]; h_in = layout["page_h_in"]
    ncols = layout["ncols"]; lines_per_col = layout["lines_per_col"]
    left_in = layout["left_in"]; right_in = layout["right_in"]
    top_in = layout["top_in"]; bottom_in = layout["bottom_in"]
    fontsize = layout["fontsize_pt"]
    line_spacing = layout["line_spacing"]

    col_gap_in = 0.18  # 열 간격(인치)

    with PdfPages(out_pdf) as pdf:
        page_no = 0
        while True:
            # 한 페이지 채울 줄들을 수집
            page_blocks: List[List[str]] = [[] for _ in range(ncols)]
            filled = 0
            # 라인 수집
            for col in range(ncols):
                for _ in range(lines_per_col):
                    try:
                        s = next(lines)
                    except StopIteration:
                        break
                    page_blocks[col].append(s)
                    filled += 1
                if filled == 0:
                    break
            if filled == 0:
                break

            fig = plt.figure(figsize=(w_in, h_in))
            fig.subplots_adjust(0,0,1,1)

            if page_no == 0 and title:
                fig.text(0.5, 0.985, title, ha="center", va="top", fontsize=max(6, fontsize+1), family="monospace")

            # 열별 텍스트 블록
            x_start_in = left_in
            col_width_in = (w_in - left_in - right_in - (ncols-1)*col_gap_in) / ncols
            y_top = 1 - (top_in / h_in)

            for c in range(ncols):
                block = "\n".join(page_blocks[c])
                if not block:
                    continue
                x_fig = (x_start_in + c*(col_width_in + col_gap_in)) / w_in
                fig.text(x_fig, y_top, block, ha="left", va="top",
                         fontsize=fontsize, family="monospace", linespacing=line_spacing)

            pdf.savefig(fig)
            plt.close(fig)
            page_no += 1


# ----------------------------- 통계/결측/일관성 체크(스트리밍) -----------------------------

class Welford:
    """온라인 평균/분산 추정."""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
    def add(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)
    def result(self) -> Tuple[int,float,float]:
        var = self.M2 / self.n if self.n > 1 else 0.0
        return self.n, self.mean, math.sqrt(var)


def _reservoir_sample_update(res: np.ndarray, k: int, stream_count: int, val: float, rng: np.random.RandomState):
    """Vitter reservoir sampling: res 크기 k. stream_count는 1부터 증가."""
    if stream_count <= k:
        res[stream_count-1] = val
    else:
        j = rng.randint(1, stream_count+1)
        if j <= k:
            res[j-1] = val


def robust_stats_and_checks(
    dta_path: str,
    out_stats_path: Optional[str] = None,
    eps_zero: float = 1e-12,
    qsample_size: int = 300_000,
) -> Dict[str, object]:
    """
    .dta 전체를 스트리밍하며 결측/통계/일관성 점검을 수행.
    - 대각선 검증(i==j ⇒ label 동일 & distance≈0)
    - distance NaN/Inf/음수, 비대각 0/극소 값
    - 비대각 분포: 평균/표준편차(정확), 분위수(저장소 샘플 근사)
    - 동일연도/교차연도 분포까지 분리
    """
    rng = np.random.RandomState(42)

    # 카운터/누적기
    total_rows = 0
    diag_rows = 0
    nan_count = 0
    inf_count = 0
    neg_count = 0

    offdiag_zero = 0
    offdiag_eps = 0

    # 레이블·연도 일관성 점검용(극소만 사용)
    diag_label_mismatch = 0
    diag_nonzero = 0

    # 분해용 Welford
    wf_off = Welford()

    # reservoir samples
    samp_off = np.empty(qsample_size, dtype=float)
    s_off_n = 0
    samp_same = np.empty(qsample_size//3, dtype=float)
    s_same_n = 0
    samp_cross = np.empty(qsample_size//3, dtype=float)
    s_cross_n = 0

    same_year_rows = 0
    cross_year_rows = 0

    # label 수 계산을 위해 대각 카운트 사용
    # 동시에 index/label 일관성 검증
    for chunk in _iter_rows_stata(
        dta_path,
        columns=["i_idx","j_idx","i_label","j_label","i_year","j_year","distance"],
        chunksize=500_000
    ):
        i = chunk["i_idx"].to_numpy(dtype=np.int64)
        j = chunk["j_idx"].to_numpy(dtype=np.int64)
        di = chunk["distance"].to_numpy(dtype=float)

        total_rows += len(chunk)

        # NaN/Inf
        nan_mask = pd.isna(di)
        inf_mask = ~np.isfinite(di) & ~nan_mask
        nan_count += int(nan_mask.sum())
        inf_count += int(inf_mask.sum())
        # 음수
        neg_count += int((di < 0).sum())

        # 대각
        dmask = (i == j)
        diag_rows += int(dmask.sum())
        if "i_label" in chunk.columns and "j_label" in chunk.columns:
            diag_label_mismatch += int((chunk.loc[dmask, "i_label"].astype(str) != chunk.loc[dmask, "j_label"].astype(str)).sum())
        # 대각 distance ~ 0 검사
        if dmask.any():
            diag_nonzero += int((np.abs(di[dmask]) > eps_zero).sum())

        # 비대각
        odmask = ~dmask
        if odmask.any():
            od = di[odmask]
            # 0/극소 개수
            offdiag_zero += int((od == 0.0).sum())
            offdiag_eps  += int((np.abs(od) <= eps_zero).sum())

            # Welford
            for x in od:
                wf_off.add(float(x))

            # 동일/교차 연도 분해
            same_m = (chunk.loc[odmask, "i_year"].astype(int).to_numpy() == chunk.loc[odmask, "j_year"].astype(int).to_numpy())
            cross_m = ~same_m
            same_year_rows += int(same_m.sum())
            cross_year_rows += int(cross_m.sum())

            # reservoir samples
            for x in od:
                s_off_n += 1
                _reservoir_sample_update(samp_off, qsample_size, s_off_n, float(x), rng)

            # same
            if same_m.any():
                od_same = od[same_m]
                for x in od_same:
                    s_same_n += 1
                    _reservoir_sample_update(samp_same, samp_same.size, s_same_n, float(x), rng)
            # cross
            if cross_m.any():
                od_cross = od[cross_m]
                for x in od_cross:
                    s_cross_n += 1
                    _reservoir_sample_update(samp_cross, samp_cross.size, s_cross_n, float(x), rng)

    # 라벨 수/기대 하삼각 수
    n_labels = diag_rows
    expected_lower = n_labels * (n_labels + 1) // 2

    # 결과 집계
    res: Dict[str, object] = {
        "total_rows": int(total_rows),
        "n_labels_via_diag": int(n_labels),
        "expected_lower_rows": int(expected_lower),
        "rows_match_lower": bool(total_rows == expected_lower),
        "nan_count": int(nan_count),
        "inf_count": int(inf_count),
        "neg_count": int(neg_count),
        "diag_rows": int(diag_rows),
        "diag_label_mismatch": int(diag_label_mismatch),
        "diag_nonzero_over_eps": int(diag_nonzero),
        "offdiag_zero": int(offdiag_zero),
        "offdiag_abs_le_eps": int(offdiag_eps),
        "same_year_rows": int(same_year_rows),
        "cross_year_rows": int(cross_year_rows),
        "eps_zero": float(eps_zero),
    }

    # 비대각 평균/표준편차
    n_off, mean_off, std_off = wf_off.result()
    res.update({
        "offdiag_count": int(n_off),
        "offdiag_mean": float(mean_off),
        "offdiag_std": float(std_off),
    })

    # 분위수(근사: reservoir)
    def _qdict(arr: np.ndarray, n_stream: int, qs=(0.0,0.01,0.05,0.25,0.5,0.75,0.95,0.99,1.0)):
        if n_stream == 0:
            return {}
        k = min(arr.size, n_stream)
        vals = np.sort(arr[:k])
        out = {}
        for q in qs:
            out[str(q)] = float(np.quantile(vals, q))
        return out

    res["offdiag_quantiles"] = _qdict(samp_off, s_off_n)
    res["sameyear_quantiles"] = _qdict(samp_same, s_same_n)
    res["crossyear_quantiles"] = _qdict(samp_cross, s_cross_n)

    # 저장
    if out_stats_path:
        os.makedirs(os.path.dirname(out_stats_path) or ".", exist_ok=True)
        with open(out_stats_path, "w", encoding="utf-8") as f:
            for k, v in res.items():
                f.write(f"{k}: {v}\n")

    return res


# ----------------------------- 메인 -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Print dense PDF from DTA and run robust stats/checks.")
    ap.add_argument("--dta", type=str, required=True, help="Path to dist_post_pca_euclidean_long.dta")
    ap.add_argument("--out-pdf", type=str, required=True, help="Output PDF path")

    # 출력 모드
    ap.add_argument("--mode", type=str, default="stride", choices=["diagonal","stride","window","all"], help="What to print")
    ap.add_argument("--stride", type=int, default=2, help="For mode=stride, keep ~1/stride of pairs (deterministic)")
    ap.add_argument("--i-min", type=int, default=None); ap.add_argument("--i-max", type=int, default=None)
    ap.add_argument("--j-min", type=int, default=None); ap.add_argument("--j-max", type=int, default=None)

    # 표시 옵션(더 빽빽하게)
    ap.add_argument("--precision", type=int, default=4, help="Float precision (significant digits)")
    ap.add_argument("--index-base", type=int, default=1, choices=[0,1], help="Print (i,j) as 0- or 1-based indices")
    ap.add_argument("--include-labels", type=str, default="no", choices=["no","yes"], help="Append [i_label vs j_label]")

    # 레이아웃(더 작은 폰트, 더 많은 열)
    ap.add_argument("--page-size", type=str, default="letter", choices=["letter","a4"])
    ap.add_argument("--top-in", type=float, default=0.35); ap.add_argument("--bottom-in", type=float, default=0.35)
    ap.add_argument("--left-in", type=float, default=0.35); ap.add_argument("--right-in", type=float, default=0.35)
    ap.add_argument("--ncols", type=int, default=5); ap.add_argument("--fontsize-pt", type=float, default=4.2); ap.add_argument("--line-spacing", type=float, default=1.0)

    # 안전장치
    ap.add_argument("--max-pages", type=int, default=200, help="If >0, hard-cap number of pages (default 200)")
    ap.add_argument("--force-full", type=str, default="no", choices=["no","yes"], help="Allow mode=all without cap (DANGEROUS)")

    # 통계/결측 체크 저장
    ap.add_argument("--stats-out", type=str, default=None, help="Write robust stats/checks to text file")
    ap.add_argument("--eps-zero", type=float, default=1e-12, help="Zero tolerance (|d|<=eps considered zero-ish)")
    ap.add_argument("--qsample-size", type=int, default=300_000, help="Reservoir sample size for quantiles (off-diag)")

    args = ap.parse_args()

    # 레이아웃 및 페이지 당 라인 계산
    layout = compute_layout(
        page_size=args.page_size,
        top_in=args.top_in, bottom_in=args.bottom_in, left_in=args.left_in, right_in=args.right_in,
        ncols=args.ncols, fontsize_pt=args.fontsize-pt if hasattr(args, 'fontsize-pt') else args.fontsize_pt,  # safeguard
        line_spacing=args.line_spacing
    )
    # (위 safeguard는 argparse 네임 충돌 방지용—실제 사용은 args.fontsize_pt)
    layout["fontsize_pt"] = args.fontsize_pt

    lines_per_page = layout["lines_per_page"]

    # 라벨 수 및 전체 하삼각 행 수 추정
    n_labels = _count_labels_via_diag(args.dta)
    total_rows_lower = n_labels * (n_labels + 1) // 2

    # 모드에 따른 예상 라인 수
    if args.mode == "diagonal":
        est_lines = n_labels
    elif args.mode == "stride":
        est_lines = max(1, total_rows_lower // max(1, args.stride))
    elif args.mode == "window":
        # 대략적 추정(상한): (i_max-i_min+1)*(j_max-j_min+1)/2
        imn = args.i_min if args.i_min is not None else 0
        imx = args.i_max if args.i_max is not None else n_labels-1
        jmn = args.j_min if args.j_min is not None else 0
        jmx = args.j_max if args.j_max is not None else n_labels-1
        wi = max(0, imx - imn + 1); wj = max(0, jmx - jmn + 1)
        est_lines = (wi * wj + 1) // 2
    else:  # all
        est_lines = total_rows_lower

    est_pages = max(1, math.ceil(est_lines / max(1, lines_per_page)))
    eprint(f"[info] labels={n_labels:,}, lower-triangle rows={total_rows_lower:,}")
    eprint(f"[info] mode={args.mode}, estimated lines≈{est_lines:,}, pages≈{est_pages:,} (lines/page={lines_per_page})")

    # 안전장치: 너무 큰 출력 경고/중단
    if args.max_pages > 0 and est_pages > args.max_pages:
        eprint(f"[warn] Estimated pages ({est_pages:,}) exceed max-pages={args.max_pages}. Output will be capped to {args.max_pages} pages.")
    if args.mode == "all" and args.force_full != "yes":
        eprint("[abort] mode=all requires --force-full yes (extremely large).")
        sys.exit(2)

    # 로버스트 통계/결측 체크 먼저 수행(증거 파일로 남김 권장)
    stats_out = args.stats_out or (os.path.splitext(args.out_pdf)[0] + "_stats.txt")
    eprint(f"[info] Running robust stats/checks → {stats_out}")
    rep = robust_stats_and_checks(
        dta_path=args.dta,
        out_stats_path=stats_out,
        eps_zero=args.eps_zero,
        qsample_size=args.qsample_size,
    )
    eprint("[ok] Stats/checks done.")

    # 라인 제너레이터
    include_labels = (args.include_labels == "yes")
    gen = lines_generator(
        dta_path=args.dta,
        mode=args.mode,
        precision=args.precision,
        index_base=args.index_base,
        stride=args.stride,
        i_min=args.i_min, i_max=args.i_max, j_min=args.j_min, j_max=args.j_max,
        include_labels=include_labels
    )

    # 최대 페이지수 제한 적용
    if args.max_pages > 0:
        cap_lines = args.max_pages * lines_per_page
        def limited_gen():
            nonlocal gen, cap_lines
            k = 0
            for s in gen:
                if k >= cap_lines:
                    break
                yield s
                k += 1
        gen_to_use = limited_gen()
        title = f"{os.path.basename(args.dta)}  [{args.mode}, stride={args.stride}]  (lines/page={lines_per_page}, cols={args.ncols}, font={args.fontsize_pt}pt)"
    else:
        gen_to_use = gen
        title = f"{os.path.basename(args.dta)}  [{args.mode}, stride={args.stride}]  (lines/page={lines_per_page}, cols={args.ncols}, font={args.fontsize_pt}pt)"

    # PDF 작성
    os.makedirs(os.path.dirname(args.out_pdf) or ".", exist_ok=True)
    eprint(f"[info] Writing PDF → {args.out_pdf}")
    write_pdf(args.out_pdf, gen_to_use, layout, title=title)
    eprint("[done] PDF creation finished.")
    eprint(f"[hint] Robust stats/checks saved: {stats_out}")


if __name__ == "__main__":
    main()
