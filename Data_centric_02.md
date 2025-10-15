아래는 질문자(해군 연구원)를 염두에 두고, **수중(수상/수중음향·PAS/PUA) 도메인**에 한정하여 *바이어스 없이, 근거와 링크가 명확한* 형태로 정리한 답변입니다. 인용은 모두 **동일·유사 주제의 공신력 있는 논문/공식 문서**에 한해 달았고, 사용하신 두 첨부 초고(ICASSP, GRSL)는 본문에 **파일 인용**으로 사실관계를 명시했습니다.

---

## 요약(Executive answers)

1. **“네트워크는 건드리지 않고 데이터를 ‘잘 요리’해서(정제/선별/캘리브레이션/증강) 성능을 끌어올리는 접근이 최근의 큰 주제인가?”**
   → 예. 전반적인 흐름은 “**Data‑Centric AI**”로 정리되며, 대표적으로 **데이터 어트리뷰션(TRAK, TracIn)**, **라벨 노이즈 정정(Confident Learning)**, **능동학습/표본 선택**, **후처리 캘리브레이션(ECE/Dirichlet/Temperature Scaling)**, **공정한 연산비용 통제(Green AI)** 같은 축이 표준화됐습니다. 음향/수중음향에서도 동일한 경향이 확인됩니다(도메인 사례는 §A 참조). ([arXiv][1])

2. **“오픈 DB에서 이미 최고치가 90%라면, ‘데이터 요리’로 80%→82% 같은 소폭 향상이 의미 있는가?”**
   → **조건부로 ‘의미 있음’**입니다. (i) *동일 추론비용/훈련비용 하에서* 달성되었는지(Compute Fairness), (ii) **정확도 외의 임무 연관 지표**(선택적 예측의 위험–커버리지, **AUGRC/AURC**, **debiased‑ECE**, NLL, **risk@coverage**)가 **유의하게 개선**되는지, (iii) **통계적 유의성(예: McNemar/부트스트랩, 여러 데이터셋에 대한 Demšar 권고 검정)**이 확보되는지를 함께 봐야 합니다. 실무 임무(탐지/식별)에서는 2pp 정확도보다 **고신뢰 구간에서의 위험 감소**가 더 큰 가치일 때가 많습니다. ([Google Research][2])

2‑1) **“SOTA(90%)를 못 이겨도 일관된 80%→82%가 가치가 있는가?”**
→ 네. **(a) 고신뢰 커버리지에서의 위험 감소(AUGRC↓, risk@80%↓), (b) 캘리브레이션(ECE↓), (c) 도메인 시프트(거리·수심·잡음) 하 일관성**을 함께 증명한다면 **충분히 기여**로 인정됩니다. 아래 **수중 도메인 실제 사례**에서 *훈련/추론비용을 고정*한 채 **≤1% 데이터만 영향기반으로 정제**했을 때 **정확도 +0.6~1.0pp, AUGRC 10~13% 상대 개선, debiased‑ECE 현저 개선**을 재현했습니다(파일 인용). 

3. **“경량화 AI(무인체계·제한 컴퓨팅) 관점의 유용 기술?”**
   → **추론비용 불변**을 전제로 한 *(i) 후처리 캘리브레이션*, *(ii) 선택적 예측(거부/확신 기반 보고)*, *(iii) 양자화(INT8)·프루닝·지식증류*, *(iv) 파라미터 효율 미세조정(Adapters/LoRA/SSF; SSF는 추론 오버헤드 0)*, *(v) AST/PaSST 같은 스펙트로그램 트랜스포머의 **Inference Patchout OFF** 운영*, *(vi) Jetson/TFLite INT8* 등이 **SWaP-C 제약 하에서 실전성**이 높습니다(세부는 §C). ([Google Research][2])

---

## A. 도메인 팩트 — “데이터 중심”이 실제로 통했는가? (수중음향 사례 모음)

* **전처리/표현만 바꿔도 성능 급상승**
  VTUAD 원천 계열(ONC)에서 파생된 선행연구는 **CQT/감마톤/멜** 등 **스펙트로그램 표현 결합**만으로 **ResNet 계열에서 95% 내외**까지 끌어올렸고, 이전 대비 **+10pp** 향상을 보고했습니다. 즉, **아키텍처 동일** 상태의 **데이터 표현/정제**가 주효했습니다. ([Research @ Flinders][3])

* **라벨 노이즈 정정의 중요성(검증 집합 포함)**
  대형 벤치마크 전반에 **평균 ≥3.3% 테스트 라벨 오류**가 있음을 대규모 크라우드 검증으로 입증—**라벨 정정만으로 모델 선정 결과가 뒤집힐 수 있음**을 보였습니다. **수중음향처럼 노이즈가 심한 레이블**에서는 ‘정확도 몇 pp’보다 **캘리브레이션 안정화**가 더 실질적으로 작전 임무에 유리합니다. ([arXiv][4])

* **영향기반(Influence) 데이터 정제·선별**
  대규모 딥러닝에서 실용적으로 쓰일 수 있는 **TRAK**(ICML)과 **TracIn**(NeurIPS)은 훈련 예제가 **개별 예측에 끼친 영향**을 근사적으로 추정, **유해 표본 삭제/다운웨이트, 우선 라벨링 타깃 선정**에 쓰입니다. 본문 **GRSL 초고(수중음향, VTUAD)**와 **ICASSP 초고(산업/감시 음향)**는 **모델 구조 고정·추론비용 동일** 조건에서, **≤1~4%**의 **미세 정제**만으로 **정확도·AUGRC·ECE**를 **일관 개선**함을 보였습니다(정량 아래 표). ([Proceedings of Machine Learning Research][5])

  * **VTUAD(멀티 시나리오 ALL, 추론비용 동일)**: **+0.6~1.0pp Acc**, **AUGRC 10~13% 상대 개선**, **debiased‑ECE 1.0pp 내외 감소**, **risk@80% ~10–12%↓**. 
  * **VTUAD(LODO/거리 시프트)**: 각 거리 시나리오 홀드아웃에서 **Acc ~+0.7–0.8pp**, **AUGRC·ECE 일관 개선**(통계 검정 포함).
  * **산업/감시 음향(동결 백본)**: **q=2–4%** 미세 프루닝에서 **정확도/매크로F1/캘리브레이션(ECE, Brier)** 동시 개선. 과도 프루닝(q≥7%)은 역효과. **능동선별(선정회수)**도 **부정 영향(negative influence)** 순위가 **불확실성(entropy) 단독**보다 우수.

* **데이터셋·운용조건의 구조화(거리·시나리오)와 최신 SOTA**
  VTUAD는 **거리 기반 시나리오(S1–S3)**가 명시된, **AIS 크로스체크**로 라벨 신뢰도를 높인 공개 코퍼스입니다. 2025년 **CATFISH**는 **학습 가능한 필터뱅크**로 **ALL 설정 96.63%**를 보고(멀티시나리오), **시나리오 결합의 난이도**를 강조합니다(이는 *데이터 구성·분할*의 중요성을 시사). 본 답변의 초점은 **SOTA 네트워크 교체 없이**도 **정제/캘리브레이션/평가규범**으로 **신뢰성**을 끌어올릴 수 있음을 *실험적으로* 보였다는 점입니다. ([arXiv][6])

* **(보충) 수중음향 벤치마크/데이터의 확대**
  ShipsEar, DeepShip, QiandaoEar22, Oceanship 등 공개 리소스가 확대되어 **표본 선택·표현/증강** 비교연구가 활발합니다. **전처리/증강(예: SpecAugment, Mixup)**은 *모델 고정* 상태에서도 효과적임이 다수 분야에서 보고되었습니다. ([ScienceDirect][7])

---

## B. “80%→82%”의 의미를 학술적으로 판단하는 프레임

1. **연산·추론 공정성(Compute Fairness) 확보**

* **훈련 총 FLOPs 통제** + **추론 토큰·옵 연동 동일(“Inference Parity”)**을 명시 보고—**Green AI** 권고. 개선이 **데이터 조치** 때문인지 **숨은 계산증가** 때문인지 분리합니다. (GRSL 초고는 PaSST Train‑Patchout에 따른 **스텝 보정**으로 FLOPs를 맞춥니다.) ([Google Research][2])

2. **임무정렬 지표와 단일 임계값 지표를 함께 제시**

* **AUGRC(post‑calib)**: 선택적 예측(거부 포함)에서 **커버리지 대비 평균 위험**—작전에서 **고신뢰 영역 안정화**를 직접 반영.
* **AURC(pre‑calib)**: 캘리브레이션과 분리된 **순수 랭킹 품질**.
* **debiased‑ECE/NLL/Brier**: **확률**의 신뢰성(오경신 감소). **온도 스케일링**(Guo et al.)/ **Dirichlet 캘리브레이션**(Kull et al.)은 **후처리**로 추론비용을 늘리지 않고 확률을 바로잡습니다. ([users.cs.fiu.edu][8])

3. **통계 유의성/일관성**

* **동일 테스트셋 상 두 분류기 비교**: **McNemar**(페어드), 또는 세션/임무 단위 **클러스터 부트스트랩**.
* **다중 데이터셋 비교**: **Demšar** 권고(윌콕슨·프리드먼+사후검정).
* **재현성**: 시드 다중화, 편향 누출 차단(선정용 SelVal vs 튜닝용 EvalVal **완전 분리**, 교차‑피팅), *편집 인덱스/해시 공개* 등의 **감사가능 절차**. (GRSL 초고 방법론·보고 양식 참조) ([Journal of Machine Learning Research][9])

> **판단 가이드:**
>
> * **정확도 +2pp**가 **AUGRC 8–12% 상대 개선** 또는 **risk@80%(단일 임계) 8–12%↓**, **ECE 1pp↓**와 동반되면 **유의미**.
> * **여러 DB/시나리오(예: 거리 LODO)에서 일관**될수록 **가치↑**—특히 실전에서는 **칼리브레이션 안정**과 **고신뢰 커버리지 유지**가 핵심 임무지표(오경보·미탐률)와 직결.

---

## C. 경량화·무인체계(제한 컴퓨팅) 대응 기술 로드맵

* **후처리 캘리브레이션**: **온도 스케일링**은 **파라미터 1개**로 ECE/NLL을 크게 낮추며 **추론 비용 0**. 필요시 **Dirichlet 캘리브레이션** 검토. ([arXiv][10])
* **선택적 예측(거부)·Risk–Coverage 운영**: **SelectiveNet/리스크‑커버리지 곡선** 기반 **커버리지 제어**—로컬 플랫폼에서 **저비용 고신뢰** 운영(불확실 사례만 저장/링크). ([ACL Anthology][11])
* **파라미터 효율 미세조정(PEFT)**: **Adapters**(추론 소폭↑), **LoRA**(추론 지연 거의 0), **SSF**(스케일‑시프트, **추론 오버헤드 0**, 합치기 가능)로 **메모리/전력** 절감. ([arXiv][12])
* **양자화/프루닝/증류**: **TFLite PTQ/INT8** 및 **QAT**는 **지연·전력↓**와 **정확도 유지** 균형이 좋음(오디오 분류 다수 보고). Jetson류(Orin/Xavier)에서 **INT8 TensorRT** 경로 권장. ([arXiv][13])
* **백본/토크나이제이션 운영**: PaSST의 **Train Patchout**은 훈련 효율↑, **Inference Patchout=OFF**로 추론 토큰 **동일**(비용 고정). **시간 해상도(hop)**를 늘려도 **추론 토큰**을 유지하는 설계가 가능(본 초고의 **CF‑FLOPs** 절차).
* **현실적 파이프라인**: (1) **백본 동결**(AST/PaSST), (2) **TRAK/TracIn**으로 **하위 0.5–1%** 정제, (3) **학습 FLOPs 정합** 후 재훈련, (4) **온도 1개**로 캘리브레이션, (5) **risk@coverage** 임계 운영, (6) **INT8 배포**.

---

## D. 질문 1–3에 대한 “링크 가능한” 근거 정리(핵심 레퍼런스)

### D‑1. 데이터 중심(정제/선별/캘리브레이션/증강) — 일반 근거

* **TRAK (ICML’23)**: 대규모 모델에 실용적인 **데이터 어트리뷰션**. 코드는 공개. ([Proceedings of Machine Learning Research][5])
* **TracIn (NeurIPS’20)**: 체크포인트‑기반 **훈련 데이터 영향 추정**. ([NeurIPS Proceedings][14])
* **Confident Learning**: **라벨 오류** 탐지·정정, 여러 벤치마크에서 **오류 ≥3.3%** 실증. ([arXiv][4])
* **캘리브레이션**: **Temperature Scaling**(ICML’17), **Dirichlet**(NeurIPS’19) — **추가 추론비용 없음**. ([arXiv][10])
* **증강(음향)**: **SpecAugment**(Interspeech’19), **Mixup**(오디오 태깅 포함) — **구조 유지** 상태의 성능·불확실성 동시 개선 보고. ([arXiv][15])
* **평가와 공정성**: **Green AI**(CACM’20) — 정확도 외에 **연산비용·탄소** 보고 권고. ([Google Research][2])

### D‑2. 수중음향/선박소음 도메인 — 데이터·방법 사례

* **VTUAD(ONC 기반)**: 거리 시나리오(S1–S3)와 AIS 기반 라벨, 2017년 조사지역/기간 요약. ([MDPI][16])
* **전처리 효과(ResNet 고정)**: CQT/감마톤/멜 결합만으로 **+10pp** 수준 향상(IEEE Access). ([Research @ Flinders][3])
* **CATFISH(2025)**: VTUAD ALL에서 **96.63%**, 시나리오 결합 난이도 강조. (※ 본 답변의 요지는 **아키텍처 교체 없이도** 신뢰성 개선 가능함) ([arXiv][6])
* **대체 공개셋**: ShipsEar, DeepShip, QiandaoEar22, Oceanship(ONC 크롤) 등. ([ScienceDirect][7])
* **GAN/증강/소샘플**: 수중 ATR에서 **데이터 증강**(GAN 포함)·소샘플/전이학습의 유효성 보고. ([MDPI][17])

### D‑3. 첨부 초고(직접 증거: 아키텍처 고정/추론비용 동일, 정제만으로 신뢰성↑)

* **GRSL 초고(수중음향·VTUAD)**:
  – **≤1% TRAK 정제**만으로 **Acc +0.6–1.0pp**, **AUGRC 10–13%↓**, **risk@80% ~9–12%↓**, **ECE 개선**.
  – **LODO(거리 시프트)**에서도 **유의 개선**(통계 검정·부트스트랩/홀름–호크버그).
  – **CF‑FLOPs & Inference Parity**를 엄격 적용(PaSST Train‑Patchout 보정 포함).
* **ICASSP 초고(산업/감시 음향·백본 동결)**:
  – **q=2–4%** 미세 정제에서 **정확도/매크로F1/캘리브레이션** 동시 개선, 과도 프루닝은 역효과.
  – **부정 영향 기반 능동선별**이 **불확실성 단독**보다 라벨링 효율↑.

---

## E. 실무 권고안(해군 운용 전제, 요약 체크리스트)

1. **데이터·분할**: 거리/해역/플랫폼별 **구조적 시프트**를 **명시 분할**(예: VTUAD S1–S3 식)로 재현. *선정용(SelVal)–튜닝/캘리브레이션용(EvalVal) 완전 분리*로 누출 차단. 
2. **미세 정제(≤1%)**: **TRAK/TracIn**으로 **유해 표본만 제거/다운웨이트**. **과대 프루닝 금지**. ([Proceedings of Machine Learning Research][5])  
3. **훈련‑추론 공정성**: **총 FLOPs 정합 + 추론 토큰 동일(패치아웃 OFF)**—개선 원인 분리. 
4. **캘리브레이션**: 온도 1개(필요시 Dirichlet 검토), **debiased‑ECE/NLL/Brier** 보고. ([arXiv][10])
5. **임무 지표**: **AUGRC/AURC + risk@coverage**를 정확도와 **함께** 보고(고신뢰 커버리지 구간 시각화). 
6. **통계 검정**: **McNemar/부트스트랩**, 다데이터셋 비교 시 **Demšar 권고** 준수. ([rasbt.github.io][18])
7. **경량 배포**: **PEFT(LoRA/SSF) + INT8 양자화** + **선택적 예측 운영**(의심 사례만 저장/전송) → **SWaP‑C/통신량** 동시 절감. ([arXiv][19])

---

## 참고문헌(주요 인용; 클릭 시 원문/공식 자료)

* **데이터 어트리뷰션**: TRAK(ICML’23), TracIn(NeurIPS’20). ([Proceedings of Machine Learning Research][5])
* **라벨 노이즈/정정**: Northcutt et al. (NeurIPS/JAIR, 2021; 벤치마크 오류 ≥3.3%). ([arXiv][4])
* **캘리브레이션**: Guo et al. (ICML’17), Kull et al. (NeurIPS’19). ([arXiv][10])
* **증강(음향)**: SpecAugment(Interspeech’19). ([arXiv][15])
* **평가 공정성**: Green AI(CACM’20). ([Google Research][2])
* **수중음향 데이터/사례**:
  – VTUAD/ONC 설명 및 활용(MDPI, 2025; ONC 포털). ([MDPI][16])
  – 전처리 결합만으로 성능↑(IEEE Access, 2022). ([Research @ Flinders][3])
  – CATFISH(SOTA, 2025 arXiv). ([arXiv][6])
  – ShipsEar/DeepShip/신규 공개셋(QiandaoEar22, Oceanship). ([ScienceDirect][7])
* **경량화/PEFT**: Adapters(2019), LoRA(2021), SSF(NeurIPS’22). ([arXiv][12])
* **TFLite/INT8·Jetson**: 공식 문서/가이드. ([arXiv][13])

---

### 부록: 첨부 초고에서의 **정량 표(발췌)** — *아키텍처/추론비용 동일, 데이터 정제만 수행*

* **VTUAD – ID(ALL)**:
  Acc **96.0→96.9**(AST), debiased‑ECE **4.6→1.3**, **AUGRC 0.055→0.049**, **risk@80% 0.082→0.073** (TRAK‑Removal@1%). 
* **VTUAD – LODO(거리 시프트)**:
  (AST, S1 홀드아웃) Acc **94.1→94.9**, ECE **5.4→4.6**, **AUGRC 0.071→0.063**. (S2/S3도 유사) 
* **산업/감시 음향(백본 동결)**:
  q=2–4%에서 **정확도/매크로F1/캘리브레이션** 동시 개선, **과대 프루닝 역효과**. **선택적 라벨링**은 **Negative Influence > Entropy**.

> 위 수치는 **훈련 FLOPs 정합·추론비용 동일(토큰 동일)** 조건에서의 **데이터 정제만**의 효과입니다(방법·누출 방지·통계 검정 자세히는 초고 본문 참조). 

---

## 결론(질문에 대한 직접 답변)

* **Q1.** 데이터 정제·선별·캘리브레이션으로 **아키텍처 고정** 상태에서 성능(특히 **신뢰성**)을 높이는 연구는 **주류**입니다(음향/수중음향 포함). ([Proceedings of Machine Learning Research][5])
* **Q2/2.1.** **SOTA 미경신**이라도, **연산 공정성**을 지키며 **AUGRC·risk@coverage·ECE**가 **유의하게** 좋아지고, **여러 데이터셋/거리 시나리오에서 일관**된다면 **충분히 가치**가 있습니다. 정확도 **+2pp**보다 **고신뢰 커버리지의 위험 감소**가 더 중요한 실전 케이스가 많습니다(첨부 초고의 VTUAD·LODO 증거 참조).
* **Q3.** 무인체계/제한 컴퓨팅에선 **PEFT+INT8**·**후처리 캘리브레이션**·**선택적 예측** 조합이 **추론비용을 늘리지 않고** 신뢰성을 끌어올리는 **가성비 최상**의 전략입니다. (Jetson/TFLite/PaSST 운용 팁은 §C) ([arXiv][19])

---

필요하시면 위 **체크리스트**와 **통계 검정 스크립트(부트스트랩·McNemar)**, **AUGRC/AURC 플로팅 노트북**, **TRAK/TracIn 파이프라인**을 정리해서 드리겠습니다.

[1]: https://arxiv.org/abs/2303.14186?utm_source=chatgpt.com "TRAK: Attributing Model Behavior at Scale"
[2]: https://research.google/blog/good-news-about-the-carbon-footprint-of-machine-learning-training/?utm_source=chatgpt.com "Good News About the Carbon Footprint of Machine ..."
[3]: https://researchnow.flinders.edu.au/en/publications/an-investigation-of-preprocessing-filters-and-deep-learning-metho "
        An investigation of preprocessing filters and deep learning methods for vessel type classification with underwater acoustic data
      \-  Research @ Flinders"
[4]: https://arxiv.org/abs/2103.14749?utm_source=chatgpt.com "Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks"
[5]: https://proceedings.mlr.press/v202/park23c/park23c.pdf?utm_source=chatgpt.com "TRAK: Attributing Model Behavior at Scale"
[6]: https://arxiv.org/html/2505.23964v1?utm_source=chatgpt.com "Acoustic Classification of Maritime Vessels using ..."
[7]: https://www.sciencedirect.com/science/article/abs/pii/S0003682X16301566?utm_source=chatgpt.com "ShipsEar: An underwater vessel noise database"
[8]: https://users.cs.fiu.edu/~sjha/class2023/Lecture8/Slides/2017TemperatureScaling.pdf?utm_source=chatgpt.com "On Calibration of Modern Neural Networks: Temperature ..."
[9]: https://www.jmlr.org/papers/volume7/demsar06a/demsar06a.pdf?utm_source=chatgpt.com "Statistical Comparisons of Classifiers over Multiple Data Sets"
[10]: https://arxiv.org/pdf/1910.12656?utm_source=chatgpt.com "Obtaining well-calibrated multiclass probabilities with ..."
[11]: https://aclanthology.org/2022.findings-acl.158.pdf?utm_source=chatgpt.com "Investigating Selective Prediction Approaches Across ..."
[12]: https://arxiv.org/abs/1902.00751?utm_source=chatgpt.com "[1902.00751] Parameter-Efficient Transfer Learning for NLP"
[13]: https://arxiv.org/html/2410.15361v1?utm_source=chatgpt.com "1 INTRODUCTION"
[14]: https://proceedings.neurips.cc/paper/2020/file/e6385d39ec9394f2f3a354d9d2b88eec-Paper.pdf?utm_source=chatgpt.com "Estimating Training Data Influence by Tracing Gradient ..."
[15]: https://arxiv.org/abs/1904.08779?utm_source=chatgpt.com "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition"
[16]: https://www.mdpi.com/2078-2489/16/8/665?utm_source=chatgpt.com "Deep Learning-Enhanced Ocean Acoustic Tomography"
[17]: https://www.mdpi.com/2072-4292/14/11/2648?utm_source=chatgpt.com "Deep Learning Models for Passive Sonar Signal ..."
[18]: https://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar/?utm_source=chatgpt.com "McNemar's test for classifier comparisons - mlxtend"
[19]: https://arxiv.org/abs/2106.09685?utm_source=chatgpt.com "LoRA: Low-Rank Adaptation of Large Language Models"
