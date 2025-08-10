#!/usr/bin/env bash
set -euo pipefail

PROC_DIR="data/processed_v7_ifpa_onset_fixed_thr2p0_AFR"
LABEL_SUFFIX="thr1.8"

TRAIN_START=2017
# 기본은 18, 짧은 구간은 12로 자동 완화
CALIB_MONTHS_LONG=18
CALIB_MONTHS_SHORT=12

INPUT_LEN=12
PATCH_SIZE=32
EMBED_CH=8
COUNTRY_EMB=8
STATIC_DROP=0.5

LOSS="focal"
FOCAL_ALPHA=0.85
FOCAL_GAMMA=2.0
H_WEIGHTS=(0.0 1.0)       # h3 집중
VAL_METRIC="h3_roc"
TH_POLICY="budget"
BUDGET=0.10

# 기본은 고정, 짧은 구간은 auto로 자동 전환
SAMPLER_H_REF_LONG=3

LR=2e-4
WD=2e-2
EPOCHS=6
PATIENCE=5
BATCH=8
NWORK=8
SEED=41
DEVICE="cuda:0"

# 검증 양성이 적으면 플랫 캘리브가 실패하므로, 기본은 platt 이지만 폴백 로직 추가
CALIBRATION_DEFAULT="platt"

SAVE_ROOT="series_v7_AFR_thr18_h3only_safe_v2_e6_s${SEED}"
mkdir -p "${SAVE_ROOT}"

# 라벨 키 점검
python - <<'PY'
import numpy as np
from pathlib import Path
p = Path("data/processed_v7_ifpa_onset_fixed_thr2p0_AFR/labels_all.npz")
z = np.load(p)
need = ["Y_1_thr1.8","Y_3_thr1.8","Ymask_1_thr1.8","Ymask_3_thr1.8"]
ok = all(k in z for k in need)
print("[OK] label keys for thr1.8 found" if ok else "[ERR] missing thr1.8 keys")
PY

END_YEARS=(2022 2021 2020 2019 2018)

# 간단한 프리플라이트: 학습 창에서 h1/h3 양성 개수 추정 → 어떤 h가 유리한지/플랫 가능여부 추정
preflight_py() {
python - "$@" <<'PY'
import sys, json, numpy as np, pandas as pd
from pathlib import Path
proc_dir, label_suffix, y0, y1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

z = np.load(Path(proc_dir)/"labels_all.npz")
import pyarrow.parquet as pq
df = pq.read_table(Path(proc_dir)/"sample_index.parquet").to_pandas()

import json as _j
meta = _j.load(open(Path(proc_dir)/"dataset_meta.json"))
dates = pd.to_datetime(meta["time_index"], format="%Y%m", utc=True)

# 학습 기간 인덱스 (누수 방지: t_end + h_max <= max)
horizons = sorted(meta["horizons"])
hmax = max(horizons)
t = np.where((dates.year>=y0)&(dates.year<=y1))[0]
t = t[t + hmax <= t.max()]
df_train = df[df.t_end.isin(t)].copy()

def suf(base,s):
    s=s.strip()
    if s=="" or s.startswith("thr"): return f"{base}_{s}" if s else base
    return f"{base}_thr{s}"

res={}
for h in horizons:
    Y = z[suf(f"Y_{h}", label_suffix)]
    M = z[suf(f"Ymask_{h}", label_suffix)].astype(bool)
    a = df_train["area_ord"].to_numpy()
    tt= df_train["t_end"].to_numpy()
    y = Y[a,tt]; m = M[a,tt]
    pos = int(((y==1)&m).sum())
    val_months = 18 if (y1-y0+1)>=3 else 12
    # 검증창 마지막 K개월
    tcut = t[t>=t.max()-(val_months-1)]
    mask_val = df["t_end"].isin(tcut)
    av = df[mask_val]["area_ord"].to_numpy()
    tv = df[mask_val]["t_end"].to_numpy()
    yv = Y[av,tv]; mv = M[av,tv]
    pos_val = int(((yv==1)&mv).sum())
    res[h] = {"pos_train":pos, "pos_val":pos_val}
print(json.dumps(res))
PY
}

run_one () {
  local END_YEAR="$1"
  local TEST_YEAR=$((END_YEAR + 1))
  local SAVE_DIR="${SAVE_ROOT}/exp_${TRAIN_START}_${END_YEAR}_eval_${TEST_YEAR}"
  mkdir -p "${SAVE_DIR}"

  # 프리플라이트로 h 별 양성 개수 확인
  local PF="$(preflight_py "${PROC_DIR}" "${LABEL_SUFFIX}" "${TRAIN_START}" "${END_YEAR}")"
  echo "[PREFLIGHT] ${PF}"

  # 기본값
  local MASK_POLICY="all"
  local CALIB_MONTHS="${CALIB_MONTHS_LONG}"
  local CALIB_MODE="${CALIBRATION_DEFAULT}"
  local MIN_K=2
  local SAMPLER="${SAMPLER_H_REF_LONG}"

  # 짧은 구간(END_YEAR<=2019) 완화
  if [[ "${END_YEAR}" -le 2019 ]]; then
    MASK_POLICY="any"
    CALIB_MONTHS="${CALIB_MONTHS_SHORT}"
    SAMPLER="auto"
    MIN_K=1
  fi

  # 프리플라이트에서 검증 양성이 매우 적으면(예: < 5) 캘리브 꺼버림
  # 또한 학습 양성=0이면 MIN_K=0, MASK_POLICY=any로 더 완화
  # (간단 파싱)
  POS_H1_TRAIN=$(python - <<PY
import json; d=json.loads('''${PF}'''); print(d.get('1',{}).get('pos_train',0))
PY
)
  POS_H3_TRAIN=$(python - <<PY
import json; d=json.loads('''${PF}'''); print(d.get('3',{}).get('pos_train',0))
PY
)
  POS_H1_VAL=$(python - <<PY
import json; d=json.loads('''${PF}'''); print(d.get('1',{}).get('pos_val',0))
PY
)
  POS_H3_VAL=$(python - <<PY
import json; d=json.loads('''${PF}'''); print(d.get('3',{}).get('pos_val',0))
PY
)

  if [[ "${END_YEAR}" -le 2019 ]]; then
    # 검증 양성 없거나 매우 적으면 calibration off
    if [[ ${POS_H1_VAL} -lt 5 && ${POS_H3_VAL} -lt 5 ]]; then
      CALIB_MODE="none"
    fi
    # 학습 양성이 둘 다 0이면 더 완화
    if [[ ${POS_H1_TRAIN} -eq 0 && ${POS_H3_TRAIN} -eq 0 ]]; then
      MIN_K=0
      MASK_POLICY="any"
      CALIB_MODE="none"
    fi
  fi

  echo
  echo "======================"
  echo " RUN  ${TRAIN_START}~${END_YEAR}  →  EVAL ${TEST_YEAR}"
  echo " SAVE ${SAVE_DIR}"
  echo " mask-policy=${MASK_POLICY}, min_k=${MIN_K}, calib=${CALIB_MODE}, sampler=${SAMPLER}, calib_months=${CALIB_MONTHS}"
  echo "======================"

  set +e
  python train_v7.py \
    --exp-mode next_year \
    --proc-dir   "${PROC_DIR}" \
    --train-start-year "${TRAIN_START}" --train-end-year "${END_YEAR}" \
    --calib-window-months "${CALIB_MONTHS}" \
    --input-len "${INPUT_LEN}" --mask-policy "${MASK_POLICY}" \
    --patch-size "${PATCH_SIZE}" --embed-ch "${EMBED_CH}" \
    --country-emb "${COUNTRY_EMB}" --static-drop "${STATIC_DROP}" \
    --loss "${LOSS}" --focal-alpha "${FOCAL_ALPHA}" --focal-gamma "${FOCAL_GAMMA}" \
    --h-weights "${H_WEIGHTS[@]}" \
    --val-metric "${VAL_METRIC}" \
    --th-policy "${TH_POLICY}" --budget "${BUDGET}" \
    --label-suffix "${LABEL_SUFFIX}" \
    --sampler-h-ref "${SAMPLER}" \
    --min-train-pos-per-country "${MIN_K}" \
    --lr "${LR}" --weight-decay "${WD}" \
    --device "${DEVICE}" --epochs "${EPOCHS}" --patience "${PATIENCE}" \
    --batch-size "${BATCH}" --num-workers "${NWORK}" \
    --calibration "${CALIB_MODE}" $( [[ "${CALIB_MODE}" == "platt" ]] && echo --save-calib ) \
    --save-preds --save-metrics --seed "${SEED}" \
    --save-dir "${SAVE_DIR}" 2>&1 | tee -a "${SAVE_DIR}/train.log"
  status=$?
  set -e

  if [[ ${status} -ne 0 ]]; then
    echo "[WARN] primary failed. Final fallback: sampler=auto, min_k=0, mask=any, calib=none"
    local SAVE_DIR_FB="${SAVE_DIR}_fallback"
    mkdir -p "${SAVE_DIR_FB}"
    python train_v7.py \
      --exp-mode next_year \
      --proc-dir   "${PROC_DIR}" \
      --train-start-year "${TRAIN_START}" --train-end-year "${END_YEAR}" \
      --calib-window-months "${CALIB_MONTHS_SHORT}" \
      --input-len "${INPUT_LEN}" --mask-policy "any" \
      --patch-size "${PATCH_SIZE}" --embed-ch "${EMBED_CH}" \
      --country-emb "${COUNTRY_EMB}" --static-drop "${STATIC_DROP}" \
      --loss "${LOSS}" --focal-alpha "${FOCAL_ALPHA}" --focal-gamma "${FOCAL_GAMMA}" \
      --h-weights "${H_WEIGHTS[@]}" \
      --val-metric "${VAL_METRIC}" \
      --th-policy "${TH_POLICY}" --budget "${BUDGET}" \
      --label-suffix "${LABEL_SUFFIX}" \
      --sampler-h-ref "auto" \
      --min-train-pos-per-country 0 \
      --lr "${LR}" --weight-decay "${WD}" \
      --device "${DEVICE}" --epochs "${EPOCHS}" --patience "${PATIENCE}" \
      --batch-size "${BATCH}" --num-workers "${NWORK}" \
      --calibration "none" \
      --save-preds --save-metrics --seed "${SEED}" \
      --save-dir "${SAVE_DIR_FB}" 2>&1 | tee -a "${SAVE_DIR_FB}/train.log"
    grep -Hn "TEST per-h metrics" "${SAVE_DIR_FB}/train.log" || true
  else
    grep -Hn "TEST per-h metrics" "${SAVE_DIR}/train.log" || true
  fi
}

for Y in "${END_YEARS[@]}"; do
  run_one "${Y}"
done

echo
echo "All done. Results under: ${SAVE_ROOT}"
grep -H "TEST per-h metrics" -n ${SAVE_ROOT}/exp_*/train.log ${SAVE_ROOT}/exp_*_fallback/train.log 2>/dev/null || true
