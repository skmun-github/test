아래 정리는 **해군 PUA(수중 수동음향)·감시/산업 음향**을 염두에 두고, *모델 구조는 그대로 두고 데이터를 잘 다뤄 성능/신뢰성을 끌어올리는* 접근을 **초보자도 따라올 수 있게** 단계별로 풀어쓴 것입니다. 내용의 “팩트”는 첨부 초고 2편(ICASSP 제출본·GRSL 투고본)을 중심으로, 관련 **공식/학술 출처**로 교차검증했습니다. 해당 초고는 본문 곳곳에 **파일 인용**으로 명시합니다.  

---

## 0) 한눈에 보는 핵심

* 요즘 한 축의 흐름은 **Data‑Centric AI**: 모델은 손대지 않고 **데이터 정제·선별·캘리브레이션**으로 **정확도＋신뢰성(ECE, Risk–Coverage)**을 **같은 추론비용**에서 개선하는 것.
* **우리 첨부 결과**(수중 VTUAD / 산업·감시 음향):

  * **VTUAD**에서 훈련셋의 **≤1%만 영향도 기반(TRAK)으로 정제**해 다시 학습하면, **정확도 +0.6~1.0%p**, **AURC(사전 보정)·AUGRC(사후 보정) 각각 약 10~13% 상대 개선**, **risk@80% 약 9~12% 감소**, **ECE 감소**—**추론비용은 동일**. 
  * **산업/감시 음향**에서도 **2~4% 소량 정제**만으로 **정확도·Macro‑F1·ECE 동시 개선**(과도 정제는 역효과). 
* **“80%→82%” 같은 소폭 향상도 의미가 있는가?**
  → **추론비용을 늘리지 않고**, **신뢰성 지표(AURC/AUGRC/ECE, risk@coverage)**가 **유의하게** 좋아지고, **여러 분할·시나리오**에서 **일관**되면 **충분히 의미 있음**. 검정은 **짝지은 비교(예: McNemar)**로.   ([rasbt.github.io][1])

---

## 1) 왜 “데이터 중심”인가? (쉽게 설명)

**문제 상황**: 수중음향은 **라벨 노이즈**(경계 오표기, 잡음, 중첩), **도메인 시프트**(거리/수심/다중경로), **컴퓨팅 제약**이 커서, 모델을 복잡하게 바꾸기보다 **데이터를 정리**하는 편이 더 안전하고 싸게 먹힙니다. 실제로 대형 벤치마크조차 **테스트셋 라벨 오류가 평균 ≥3.3%**임이 보고되었습니다. ⇒ “정확도 1~2%p”만 보고 모델을 갈아치우다 **오판**할 수 있다는 뜻. ([datasets-benchmarks-proceedings.neurips.cc][2])

**해법의 큰 줄기**

* **정제/선별**: 어떤 훈련 샘플이 해로운지 **영향도**로 찾아 조금만(≤1%) 걷어냅니다. (TRAK/TracIn) ([Proceedings of Machine Learning Research][3])
* **캘리브레이션**: 마지막에 **온도 1개**만 맞춰 **확률의 신뢰성(ECE/NLL)**을 교정합니다. (Temperature Scaling; 필요 시 Dirichlet) ([Proceedings of Machine Learning Research][4])
* **임무정렬 평가**: 단일 임계 정확도 대신 **Risk–Coverage 곡선**(AURC/AUGRC)을 봅니다. 커버리지를 줄이는 대신 **평균 위험**을 낮추는 **선택적 예측**이 핵심. ([arXiv][5])
* **비용 공정성**: **학습 FLOPs 맞추고, 추론 토큰/연산을 동일**하게(**Inference Parity**) 두고 비교해야 개선의 원인이 **데이터 조치**인지 분리됩니다. (Green AI 권고)  ([ACM Digital Library][6])

---

## 2) 첨부 초고 2편—무엇을 어떻게 했고, 무엇이 좋아졌나?

### 2‑A) ICASSP 제출본(산업/감시 음향): **동결 백본 + 가벼운 헤드 + TracIn**

* **무엇을 했나**: 사전학습 오디오 백본(예: PaSST/AST)은 **동결**, 얕은 헤드만 학습. **헤드 공간에서 TracIn**을 써서
  **(i) 해로운 훈련 창 소량(q≈2~4%) 정제**, **(ii) 라벨 예산이 빡빡할 때 부정영향(negative influence)** 기반 **능동선별**. 
* **왜 가능한가**: TracIn은 **체크포인트별 기울기 내적**을 합산해 **특정 훈련 예제가 검증 예제에 주는 영향**을 근사합니다(헤드‑온리라 안정·저비용).  ([NeurIPS Proceedings][7])
* **무엇이 좋아졌나**: 두 데이터셋에서 **정확도·Macro‑F1 상승**과 **ECE/Brier 개선**, **능동선별**도 **불확실성(Entropy) 단독**보다 **부정영향 또는 영향+불확실성 결합**이 상위. (표·수치 다수) 

### 2‑B) GRSL 투고본(VTUAD, 해군 도메인): **TRAK 정제 + 신뢰성 우선 + 비용공정**

* **셋업**: 수중 **VTUAD**(5클래스; 거리 시나리오 S1–S3·ALL)에서 **검증 전용 영향도(TRAK)**로 **≤1%만 제거/다운웨이트**, **총 학습 FLOPs 일치(CF‑FLOPs)**, **추론동등(Inference Parity)** 고정, **AUGRC(사후)·AURC(사전)·debiased‑ECE** 중심 평가. 
* **ID(ALL) 결과(예시)**: **AST Full‑FT**

  * **정확도 96.0→96.9%**, **AURC 0.078→0.068**, **AUGRC 0.055→0.049**, **risk@80% 0.082→0.073**, **debiased‑ECE 4.6→1.3%**. **추론비용 증가는 0**. 
* **LODO(거리 시프트) 결과**: S1/S2/S3 **모두**에서 **정확도·AURC/AUGRC·ECE** 동반 개선(**risk@80%≈9~12%↓**). 즉, **거리 분포가 바뀌어도 개선 유지**. 
* **모델/튜닝**: **AST·PaSST**(Patchout은 **학습에만**, **추론은 풀토큰**) + **PEFT(Adapters/LoRA/SSF; SSF는 추론 오버헤드 0)**.  ([isca-archive.org][8])

> 참고: VTUAD 최신 보고(CATFISH)는 **ALL에서 96.63%**를 보고—**멀티시나리오가 현실적**이면서 난도가 높음을 강조합니다(우리 논지는 “**모델 안 바꾸고도** 신뢰성 개선 가능”). ([arXiv][9])

---

## 3) 용어·지표를 **그림 없이 쉽게** 이해하기

* **영향도(Influence)** = *“이 훈련 샘플을 조금 바꾸면, 저 검증 샘플의 손실이 얼마나 변하나?”*

  * **TracIn**: 학습 도중 저장한 **여러 체크포인트의 기울기**를 써서 **두 샘플의 ‘방향이 같은지’**(도움/방해)를 빠르게 재봅니다.  ([NeurIPS Proceedings][7])
  * **TRAK**: 대규모 모델에도 **스케일**되게 만든 **데이터 어트리뷰션** 방법. **소수 모델만**으로도 정확히 근사. 코드 공개. ([arXiv][10])
* **캘리브레이션** = *“모델이 0.8이라 말할 때 실제 정답률이 80%인가?”*

  * 실무에선 **온도 1개(Temperature Scaling)**로 많은 경우 충분히 개선. 필요 시 **Dirichlet**(다중클래스) 고려. ([Proceedings of Machine Learning Research][4])
  * **ECE/NLL/Brier**로 **확률의 신뢰성**을 수치화. (ICASSP/GRSL 모두 활용)
* **Risk–Coverage 곡선** = *“덜 확실한 샘플은 보류(커버리지↓), 남긴 샘플에서 평균 오류(리스크)를 얼마나 줄였나?”*

  * **AURC(사전 보정)**: 순수 **랭킹 품질**. **AUGRC(사후 보정)**: **보정 후** 전체 임계들을 통틀어 본 **평균 위험**. 낮을수록 좋음.  ([arXiv][5])
* **비용 공정성(Compute‑Fairness) & 추론동등** = *“학습 총 FLOPs·추론 토큰이 같은 조건에서만 비교하자.”*

  * **Green AI**가 권고하는 **효율 보고/통제** 원칙. GRSL은 **PaSST의 Train‑Patchout은 학습에만** 쓰고 **추론은 풀토큰**으로 맞춰 **동등 비용**을 보장.  ([ACM Digital Library][6])

---

## 4) “이미 최고치 90%가 있다면, **80%→82%**가 의미 있나?”—**판단 절차**

### 4‑1) **비용을 먼저 고정**

* **같은 추론 연산/지연**(토큰·연산 동일), **학습 총 FLOPs**도 **매칭**했는지 확인. (GRSL의 **CF‑FLOPs & Inference Parity** 참조) 

### 4‑2) **정확도만 보지 말고 신뢰성도 본다**

* **AURC/AUGRC, ECE, risk@coverage**가 함께 **내렸는가**?

  * 예) GRSL ID(ALL)에서 **AUGRC 0.055→0.049**, **ECE 4.6→1.3%**, **risk@80% 0.082→0.073**. **추론비용은 동일**. 

### 4‑3) **통계적으로 유의한가? (짝지은 비교 권장)**

* 같은 테스트 샘플에 대해 **두 모델의 정·오답 교차표**를 만들고 **McNemar**로 유의성 판단. (대형 딥러닝처럼 **반복 재학습이 비싸면** 특히 권장) ([PubMed][11])
* **샘플수 감지력(대략계산)**: 정확도 80% 근방에서 **±1%p 오차한계(95% CI)**를 원하면
  (\ p(1-p)\left(\tfrac{1.96}{0.01}\right)^2 = 0.8\cdot0.2\cdot 196^2 \approx 0.16\cdot 38416 \approx 6{,}147\ )
  즉 **약 6,147개** 테스트 표본이 필요합니다(독립 근사). *짝지은* 검정에선 **b,c(상호불일치)** 규모가 실제 검정력을 좌우합니다.
* **여러 분할·시나리오(예: LODO)**에서 **일관된 개선**이면 실무적 설득력 ↑. (GRSL LODO 전체에서 유지) 

> **결론**: **SOTA=90%를 못 이겨도**, **추론비용 동일** 조건에서 **AUGRC/ECE/risk@coverage**가 **유의하게** 좋아지고 **다중 시나리오 일관성**이 있으면 **충분히 의미 있음**.

---

## 5) **경량화(무인체계) 관점의 실전 레시피**

> **원칙**: *추론비용은 유지*하고 **캘리브레이션·선택적 예측**으로 **신뢰구간**을 정리한 뒤, **양자화/PEFT**로 배포 효율을 끌어올립니다.

1. **백본 동결 + 얕은 헤드** → **영향도 계산 안정·저비용**, 배포도 단순. (ICASSP/GRSL 공통)
2. **PEFT**: **Adapters/LoRA/SSF**—특히 **SSF는 추론 시 합치기(merge)로 오버헤드 0**. 
3. **양자화**:

   * **TFLite INT8(Post‑Training Quantization)**—모델 크기↓·지연/전력↓, 대표 데이터로 스케일 보정. ([TensorFlow][12])
   * **TensorRT INT8**—대량 배치/엔비디아 엣지에서 효과적, 필요 시 **QAT**로 정확도 보전. ([NVIDIA Docs][13])
4. **선택적 예측 운영**: **Risk–Coverage**로 **고신뢰 커버리지**를 지정(의심 사례만 저장/전송) → **SWaP‑C·통신량** 동시 절감. ([arXiv][5])
5. **PaSST Patchout 사용법**: **학습만 Patchout**, **추론은 Full**(토큰 동일) → **공정 비교·예측 일관성**.  ([isca-archive.org][8])

---

## 6) 초보자를 위한 **실험 체크리스트** (체크박스처럼 사용)

* [ ] **분할/누출통제**: **SelVal(선정/정제)** ↔ **EvalVal(튜닝/캘리브레이션)** **완전 분리**, LODO(거리 시프트) 평가 포함. 
* [ ] **정제 예산**: **0.5–1%**부터—**1% Removal**이 대체로 최적, **과도(≥2–4%)**는 경계.  
* [ ] **비용 공정성**: **CF‑FLOPs**로 학습 스텝 조정, **Inference Parity**로 추론 토큰/연산 동일화. 
* [ ] **지표 세트**: **정확도/ROC** 외에 **AURC(사전), AUGRC(사후), debiased‑ECE/NLL/Brier, risk@coverage** 필수.
* [ ] **통계**: **McNemar**(짝지은 비교) 또는 **세션‑클러스터 부트스트랩**, 다중 비교 시 **Holm–Hochberg**.  ([PubMed][11])
* [ ] **배포**: **온도 1개**로 보정→**INT8 양자화**(**TFLite/TensorRT**)→필요 시 **PEFT**·지식증류 추가. ([Proceedings of Machine Learning Research][4])

---

## 7) 꼭 참고할 **원문/공식 링크**

* **영향도 기반 데이터 어트리뷰션**: **TRAK**(ICML’23) 원문/코드. ([Proceedings of Machine Learning Research][3])
* **TracIn**(NeurIPS’20) 원문. ([NeurIPS Proceedings][7])
* **캘리브레이션**: Temperature Scaling(ICML’17), Dirichlet Calibration(NeurIPS’19). ([Proceedings of Machine Learning Research][4])
* **선택적 예측·Risk–Coverage**: Selective Classification(2017), **SelectiveNet**(ICML’19). ([arXiv][5])
* **Green AI**(CACM’20). ([ACM Digital Library][6])
* **PaSST Patchout(학습 전용)**·**AST**(오디오 트랜스포머). ([isca-archive.org][8])
* **VTUAD 최신 보고(CATFISH 96.63%)**. ([arXiv][9])
* **라벨 오류 3.3%+**(NeurIPS’21 D&B). ([datasets-benchmarks-proceedings.neurips.cc][2])
* **양자화**: TFLite PTQ, TensorRT INT8/QAT. ([TensorFlow][12])

---

## 8) 부록—첨부 초고의 **정량 예시** (팩트만 발췌)

* **VTUAD(ALL, 동일 추론비용)**: **AST** 기준
  **Acc 96.0→96.9%**, **debiased‑ECE 4.6→1.3%**, **AURC 0.078→0.068**, **AUGRC 0.055→0.049**, **risk@80% 0.082→0.073** *(TRAK‑Removal@1%)*. 
* **VTUAD(LODO, 거리 시프트)**: 세 홀드아웃 전부에서 **Acc·AURC/AUGRC·ECE** **동반 개선**, **risk@80% ≈9–12%↓**. 
* **산업/감시 음향**: **q=2–4% 소량 정제**에서 **정확도·Macro‑F1 상승 + ECE/Brier 감소**, **과도 정제(q≥7%)**는 역효과. **능동선별**은 **부정영향 또는 영향+엔트로피 결합 > 엔트로피 단독**. 

---

### 마무리

**해군 관점**에서는 “**동일 추론비용**에서 **고신뢰 구간의 위험**을 낮추는가?”가 제일 중요합니다. 위의 **단계별 절차**(≤1% 영향도 정제 → CF‑FLOPs+Inference Parity로 재학습 → 온도 1개 보정 → AURC/AUGRC·risk@coverage·ECE 보고 → McNemar 검정)대로 진행하면, **SOTA를 ‘정확도’로 못 이겨도** **운용상 더 안전·일관된 시스템**을 만들 수 있습니다. 필요하시면 위 절차에 맞춘 **검정/CI 스크립트**와 **AURC/AUGRC 플로팅 노트북**도 정리해 드리겠습니다.

[1]: https://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar/?utm_source=chatgpt.com "McNemar's test for classifier comparisons - mlxtend"
[2]: https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/f2217062e9a397a1dca429e7d70bc6ca-Paper-round1.pdf?utm_source=chatgpt.com "Pervasive Label Errors in Test Sets Destabilize Machine ..."
[3]: https://proceedings.mlr.press/v202/park23c/park23c.pdf?utm_source=chatgpt.com "TRAK: Attributing Model Behavior at Scale"
[4]: https://proceedings.mlr.press/v70/guo17a/guo17a.pdf?utm_source=chatgpt.com "On Calibration of Modern Neural Networks"
[5]: https://arxiv.org/pdf/1705.08500?utm_source=chatgpt.com "Selective Classification for Deep Neural Networks"
[6]: https://dl.acm.org/doi/10.1145/3381831?utm_source=chatgpt.com "Green AI | Communications of the ACM"
[7]: https://proceedings.neurips.cc/paper/2020/file/e6385d39ec9394f2f3a354d9d2b88eec-Paper.pdf?utm_source=chatgpt.com "Estimating Training Data Influence by Tracing Gradient ..."
[8]: https://www.isca-archive.org/interspeech_2022/koutini22_interspeech.pdf?utm_source=chatgpt.com "Efficient Training of Audio Transformers with Patchout"
[9]: https://arxiv.org/html/2505.23964v1?utm_source=chatgpt.com "Acoustic Classification of Maritime Vessels using ..."
[10]: https://arxiv.org/abs/2303.14186?utm_source=chatgpt.com "TRAK: Attributing Model Behavior at Scale"
[11]: https://pubmed.ncbi.nlm.nih.gov/9744903/?utm_source=chatgpt.com "Approximate Statistical Tests for Comparing Supervised ..."
[12]: https://www.tensorflow.org/model_optimization/guide/quantization/post_training?utm_source=chatgpt.com "Post-training quantization - Model optimization"
[13]: https://docs.nvidia.com/deeplearning/tensorrt/10.9.0/inference-library/work-quantized-types.html?utm_source=chatgpt.com "Working with Quantized Types — NVIDIA TensorRT ..."
