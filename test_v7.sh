#!/usr/bin/env bash
set -euo pipefail

# ========= 사용자 경로/환경 =========
DEVICE="cuda:0"
SEED=41
NWORKERS=8
BS=8
EPOCHS=6
CALIB_WIN=18
TRAIN_Y0=2017
TRAIN_Y1=2022

# ========= 원천 데이터 경로 =========
NPZ="data/blacksea_tensor_log1p.npz"
META="data/blacksea_tensor_log1p.meta.json"
SPARSE="data/blacksea_sparse.pt"
PRICE="data/raw/CPIs.csv"
CROPS="data/raw/Crops_and_livestock_products.csv"
GDP="data/raw/Value_of_Agricultural_Production.csv"
WDI="data/raw/WDI_wo_CPI_related.xlsx"
RAW="data/raw"

# ========= 출력 루트 =========
OUT_ROOT="data/processed_v7_ifpa_onset_fixed_thr2p0"         # 빌드 루트(날짜 하위폴더 생성)
OUT_AFR="data/processed_v7_ifpa_onset_fixed_thr2p0_AFR"      # AFR 서브셋 폴더(최종 PROC-DIR)

# ========== 1) 데이터셋 빌드(필요시) + AFR 추출 ==========
echo "[STEP 1] Build dataset (if needed) & extract AFR subset ..."
NEED_BUILD=0
if [[ ! -d "${OUT_AFR}" ]] || [[ ! -f "${OUT_AFR}/labels_all.npz" ]]; then
  NEED_BUILD=1
fi

if [[ "${NEED_BUILD}" -eq 1 ]]; then
  echo " -> building full dataset into ${OUT_ROOT} ..."
  python -u build_dataset_v7.py \
    --npz          "${NPZ}" \
    --meta-json    "${META}" \
    --sparse-pt    "${SPARSE}" \
    --price-csv    "${PRICE}" \
    --crops-csv    "${CROPS}" \
    --gdp-csv      "${GDP}" \
    --wdi-xlsx     "${WDI}" \
    --raw-dir      "${RAW}" \
    --out-dir      "${OUT_ROOT}" \
    --window-len   12 \
    --horizon      1 3 \
    --scaler       robust \
    --month-enc    cyclical \
    --surge-mode   ifpa \
    --ifpa-gamma   0.6 \
    --ifpa-thr     2.0 \
    --multi-thr    1.8,2.0,2.5 \
    --label-timing onset \
    --min-duration 2 \
    --refractory   2 \
    --baseline-mode  fixed \
    --baseline-years 2000:2018 \
    --vintage-interp ffill \
    --cv-stride    0 \
    --val-year     2022 \
    --use-sparse \
    --verbose
else
  echo " -> dataset already exists at ${OUT_AFR}, skipping build."
fi

# 최신 날짜 폴더 찾기
LATEST_DATE=$(ls -1 "${OUT_ROOT}" | grep -E '^[0-9]{8}$' | sort | tail -n 1)
if [[ -z "${LATEST_DATE}" ]]; then
  echo "ERROR: cannot find date subfolder under ${OUT_ROOT}"
  exit 1
fi
echo " -> latest date under ${OUT_ROOT}: ${LATEST_DATE}"

# AFR 추출(필요시만)
if [[ ! -f "${OUT_AFR}/labels_all.npz" ]]; then
  echo " -> extracting AFR subset to ${OUT_AFR} ..."
  python -u extract_subset_v7.py \
    --in-dir  "${OUT_ROOT}/${LATEST_DATE}" \
    --out-dir "${OUT_AFR}"
fi

# ========== 2) 라벨 키 확인(참고 출력) ==========
echo "[STEP 2] Check label keys available in ${OUT_AFR}/labels_all.npz ..."
python - <<'PY'
import numpy as np, sys, os
p = "data/processed_v7_ifpa_onset_fixed_thr2p0_AFR/labels_all.npz"
if not os.path.exists(p):
    print("!! labels_all.npz not found:", p); sys.exit(1)
z = np.load(p)
keys = sorted([k for k in z.files if k.startswith("Y_") or k.startswith("Ymask_")])
print("\nAvailable label keys:")
for k in keys: print("  ", k)
# 간단 매핑 힌트
have = set(keys)
for s in ["thr1.8","thr2.0","thr2","thr2.5"]:
    y1 = f"Y_1_{s}"; m1 = f"Ymask_1_{s}"
    ok = (y1 in have) and (m1 in have)
    print(f"suffix '{s}':", "OK" if ok else "NOT FOUND")
PY
echo

# ========== 3) 6-epoch × 3 실험 ==========
PROC_DIR="${OUT_AFR}"
TS=$(date +%Y%m%d_%H%M%S)

# 공통 하이퍼
COMMON_ARGS=(
  --exp-mode next_year
  --proc-dir   "${PROC_DIR}"
  --train-start-year "${TRAIN_Y0}" --train-end-year "${TRAIN_Y1}"
  --calib-window-months "${CALIB_WIN}"
  --input-len 12
  --patch-size 32 --embed-ch 8 --country-emb 8 --static-drop 0.5
  --loss focal --focal-gamma 2.0
  --h-weights 0.0 1.0
  --val-metric h3_roc
  --th-policy budget --budget 0.10
  --sampler-h-ref 3
  --min-train-pos-per-country 2
  --lr 2e-4 --weight-decay 2e-2
  --device "${DEVICE}" --epochs "${EPOCHS}" --patience 5 --batch-size "${BS}" --num-workers "${NWORKERS}"
  --save-preds --save-metrics --seed ${SEED}
)

# ----- EXP A: thr2(=2.0) + Isotonic + any -----
SAVE_A="ckpt_v7_AFR_expA_thr2_iso_any_${TS}"
mkdir -p "${SAVE_A}"
python -u train_v7.py \
  "${COMMON_ARGS[@]}" \
  --mask-policy any \
  --label-suffix 'thr2' \
  --focal-alpha 0.87 \
  --calibration isotonic --save-calib \
  --save-dir "${SAVE_A}" | tee "${SAVE_A}/train.log"

# ----- EXP B: thr2.5 + Platt + any + country-emb 12 / drop 0.6 / lr 3e-4 / wd 1e-2 -----
SAVE_B="ckpt_v7_AFR_expB_thr25_platt_any_${TS}"
mkdir -p "${SAVE_B}"
python -u train_v7.py \
  --exp-mode next_year \
  --proc-dir   "${PROC_DIR}" \
  --train-start-year "${TRAIN_Y0}" --train-end-year "${TRAIN_Y1}" \
  --calib-window-months "${CALIB_WIN}" \
  --input-len 12 \
  --mask-policy any \
  --patch-size 32 --embed-ch 8 --country-emb 12 --static-drop 0.6 \
  --loss focal --focal-alpha 0.90 --focal-gamma 2.0 \
  --h-weights 0.0 1.0 \
  --val-metric h3_roc \
  --th-policy budget --budget 0.10 \
  --label-suffix 'thr2.5' \
  --sampler-h-ref auto \
  --min-train-pos-per-country 3 \
  --lr 3e-4 --weight-decay 1e-2 \
  --device "${DEVICE}" --epochs "${EPOCHS}" --patience 5 --batch-size "${BS}" --num-workers "${NWORKERS}" \
  --save-preds --save-metrics --seed ${SEED} \
  --calibration platt --save-calib \
  --save-dir "${SAVE_B}" | tee "${SAVE_B}/train.log"

# ----- EXP C: thr1.8 + Platt + mask-policy=all -----
SAVE_C="ckpt_v7_AFR_expC_thr18_platt_all_${TS}"
mkdir -p "${SAVE_C}"
python -u train_v7.py \
  "${COMMON_ARGS[@]}" \
  --mask-policy all \
  --label-suffix 'thr1.8' \
  --focal-alpha 0.87 \
  --calibration platt --save-calib \
  --save-dir "${SAVE_C}" | tee "${SAVE_C}/train.log"

echo
echo "Done. 결과 폴더:"
echo "  - ${SAVE_A}"
echo "  - ${SAVE_B}"
echo "  - ${SAVE_C}"
echo "빠른 비교:  grep -H \"TEST per-h metrics\" */train.log"
