# dist_post_pca_euclidean_long.dta — 사용 설명 (경제학자 초보자용)

이 파일은 **PCA(2차원) 임베딩 후** **국가-연도 vs 국가-연도** 간의 **유클리드 거리**를 담은 표입니다.
즉, V‑Dem 지표들을 전처리(결측 중앙값 대체, 전역 z‑score 표준화)한 뒤 PCA로 2차원에 투영하고,
그 좌표 간 거리를 계산한 값입니다.

## 1) 파일 구조(변수 설명)

- `i_country`, `j_country` : 국가 식별자(대개 ISO‑3 텍스트 코드, 예: KOR, USA).  
- `i_year`, `j_year` : 연도(예: 1993, 2008).  
- `distance` : **PCA 2D 좌표** 상의 유클리드 거리(0 이상 실수).  
- 보조 필드
  - `i_label`, `j_label` : `"<국가>-<연도>"` 형식(예: `KOR-1993`).  
  - `i_idx`, `j_idx` : 원 정사각 행렬에서의 0‑기반 인덱스.  
- 행 구조: 기본은 **하삼각(lower)**만 포함 (`i_idx <= j_idx`).  
  - 같은 국가-연도 쌍이면 거리 0 (`i_idx==j_idx`).  
  - 반대 방향(`j,i`)은 중복 제거되어 없습니다.

> 주의: 이 거리는 **원 특징 공간**의 거리가 아니라 **PCA 2D 투영** 상의 거리입니다.
시각화/군집 해석에는 유용하나, 엄정한 정량 비교가 필요하면 원공간(또는 더 많은 차원)의 거리도 참고하세요.

## 2) Stata에서 여는 법

1) Stata 실행 → 아래를 그대로 입력
```stata
clear
use "C:\path\to\dist_post_pca_euclidean_long.dta", clear
describe
count
list in 1/5
```

2) 변수 확인
```stata
codebook i_country i_year j_country j_year distance
```

## 3) 기본 조회 예시 (step-by-step)

### 3.1 두 시점 간 거리 조회
예: **대한민국 1990** 과 **미국 1990** 간 거리
```stata
list distance i_label j_label if i_country=="KOR" & i_year==1990 & j_country=="USA" & j_year==1990, noobs
```

### 3.2 특정 노드(국가-연도)의 최근접 이웃 Top‑10
예: **대한민국 1990** 기준으로 가장 가까운 10개 이웃
```stata
preserve
keep if i_country=="KOR" & i_year==1990
sort distance
list j_country j_year distance in 1/10, noobs
restore
```

### 3.3 같은 연도 내에서만 비교
예: 1990년 동시점 국가 간 거리
```stata
keep if i_year==1990 & j_year==1990         // 같은 연도만
drop if i_idx==j_idx                        // 자기자신 제거
sort distance
list i_country j_country distance in 1/10, noobs
```

### 3.4 특정 국가의 연도별 동시점 평균거리
예: 대한민국의 각 연도에서 **동일 연도 내** 평균거리
```stata
preserve
keep if i_country=="KOR" & i_year==j_year   // 동시점만 유지
bysort i_year: egen mean_d = mean(distance)
bysort i_year: keep if _n==1
sort i_year
list i_year mean_d, noobs
restore
```

### 3.5 한 국가-연도의 “가장 가까운 k개”를 테이블로 저장(엑셀)
```stata
preserve
keep if i_country=="KOR" & i_year==1990
sort distance
keep in 1/10
export excel using "nearest_KOR_1990.xlsx", firstrow(variables) replace
restore
```

### 3.6 중복/대칭 관련 주의
- 기본은 **하삼각**이므로 반대 방향(`j→i`) 행은 존재하지 않습니다.
- 자기자신(`i_idx==j_idx`)은 거리 0이며, 필요하면 제거하세요.
```stata
drop if i_idx==j_idx
```

## 4) 자주 묻는 질문

**Q1. 왜 '연도'가 따로 변수로 들어있나요? 라벨에도 연도가 있는데요.**  
A. 필터/집계를 쉽게 하도록 분리했습니다. `i_label`의 연도와 `i_year`는 동일합니다.

**Q2. 왜 어떤 분석에서 연도-불일치가 나오나요?**  
A. `i_year==j_year` 조건을 넣지 않으면 **교차연도 간 거리**도 포함됩니다. 용도에 맞게 필터하세요.

**Q3. Excel로 바로 내보낼 수 있나요?**  
A. 네, `export excel using "file.xlsx", firstrow(variables) replace` 명령을 사용하세요.

## 5) (선택) DTA가 정말 '국가-연도 vs 국가-연도'인지 검증하기

동봉된 파이썬 검증 스크립트 `validate_postpca_year_dta.py`를 사용하세요.

### 5.1 CSV 기준 검증 (정확)
```bash
python validate_postpca_year_dta.py \
  --dta ./vdem_outputs_decade/dist_post_pca_euclidean_long.dta \
  --csv ./vdem_outputs_decade/dist_post_pca_euclidean.csv.gz \
  --tol 1e-9
```

### 5.2 .pt 재계산 기반 검증 (샘플링 가능)
```bash
python validate_postpca_year_dta.py \
  --dta ./vdem_outputs_decade/dist_post_pca_euclidean_long.dta \
  --pt  V-Dem-CY-FullOthers-v15_csv/V-Dem-CY-Full+Others-v15.pt \
  --start-year 1980 --sample 50000 --tol 1e-7
```

성공하면 `.dta`가
- (a) **CSV의 post‑PCA 거리**와 동일하고,
- (b) **.pt에서 재계산한 PCA 2D 거리**와도 허용오차 내에서 일치함을 확인할 수 있습니다.
