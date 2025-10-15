제목: Re: 질문 주신 3가지—데이터 중심 접근의 위상, ‘데이터 요리’ 성과의 의미, 경량화 방향

안녕하세요.
보내주신 질문에 대해, 제가 공유드렸던 2개 초고(① ICASSP 제출본: 산업/감시 음향 데이터 큐레이션·액티브 셀렉션 레시피, ② IEEE GRSL 투고본: VTUAD에서 영향도 기반 신뢰성 향상 프로토콜)를 바탕으로 최근 문헌을 교차검증하여 정리했습니다.  

---

## 1) “네트워크는 고정하고 데이터를 잘 다뤄 성능을 올리는” 접근이 최근 큰 흐름인가?

**그렇습니다. ‘데이터 중심 AI(Data‑Centric AI)’와 데이터셋/벤치마크 연구가 최근 수년간 명확한 축으로 자리 잡았습니다.**

* NeurIPS는 2020년부터 **Datasets & Benchmarks** 트랙을 독립적으로 신설했고, 이후로도 규모가 커지며 데이터 품질·도큐멘테이션·재현성 기준을 엄격히 명시합니다. ([NeurIPS][1])
* Andrew Ng는 **데이터 중심 AI**를 “코드는 고정하고 데이터를 체계적으로 엔지니어링하는 규율”로 정의하며, 오늘날 다수 응용에서 **모델 구조보다 데이터 공학이 더 큰 한계요인**임을 강조합니다. ([IEEE Spectrum][2])
* 방법론 측면에서도,

  * **데이터 기여도/영향도 추적**(TRAK, TracIn, Influence Functions)이 대규모 모델에서도 실용적으로 쓰이며, 데이터 수정·선별의 ‘근거’를 제공합니다. ([Proceedings of Machine Learning Research][3])
  * **라벨 노이즈 추정과 데이터 정제**(Confident Learning / Cleanlab)가 범용 툴로 자리잡았습니다. ([Google Research][4])
  * **부분집합/예산 선택**(GLISTER 등)으로 학습 효율·강건성을 동시에 노리는 연구가 정례화되어 있습니다. ([dblp.org][5])

저희 **GRSL 투고본**은 이 흐름을 수중음향(VTUAD) 맥락에 옮겨, **검증 전용 영향도 기반 큐레이션→동일 학습 FLOPs(Compute‑Fairness)·동일 추론비용 유지→AURC/AUGRC·ECE 중심의 신뢰성 지표 향상**을 일관되게 보이는 프로토콜을 제안합니다.   또한 음향 백본으로 **AST·PaSST** 같은 강력한 스펙트로그램 트랜스포머를 “구조 고정” 상태로 비교합니다.   (AST, PaSST 원문은 각각 다음을 참고해 주세요. ([ISCA Archive][6]))

---

## 2) 오픈 DB에서 이미 최고치가 높게 형성된 상황에서, **데이터 조리(정제·선별)**로 소폭 향상은 의미가 있는가?

**핵심은 ‘무엇을, 어떤 비용 하에서, 얼마나 안정적으로’ 개선하는가**입니다. 정확도(Accuracy) 단일 수치만 보지 말고, **신뢰성(reliability)과 비용**을 함께 보셔야 합니다.

### 2‑A. “무엇을” — 정확도 외 **신뢰성 지표**

* **Risk–Coverage 관점**: 선택적 예측(selective prediction)에서는 **커버리지를 낮추며(불확실한 샘플 보류)** 위험(오류)을 줄이는 trade‑off가 핵심입니다. 단일 임계치 성능보다 **곡선 자체(AURC, AUGRC)**가 더 타당한 목표가 됩니다. (Selective Classification/SelectiveNet 문헌) ([NeurIPS Proceedings][7])
* **보정(Calibration)**: ECE/NLL/Brier로 **확률의 신뢰도**를 계량화합니다. 저희 GRSL 프로토콜은 **debiased‑ECE(동량 빈)**와 **온도보정(Temperature Scaling)**을 표준화하여, **AURC(보정 전)·AUGRC(보정 후)**를 함께 보고합니다. 

> **결론적으로**: **정확도가 같아도**, **AUGRC↓(보정 후 위험 평균 감소), AURC↓(랭킹 품질 개선), ECE↓(확률 일치)**가 **동일 추론비용** 하에서 함께 내려간다면 **의미 있는 개선**입니다. 저희 GRSL 결과는 **편집 ≤1%**만으로 ID/LODO 전반에서 **AURC/AUGRC·ECE가 일관되게 개선**됨을 보입니다. 

### 2‑B. “어떤 비용 하에서” — **Compute‑Fairness & 추론동등성**

**Green AI**의 권고처럼 **학습 FLOPs와 추론비용을 통제/보고**해야 ‘공정한’ 비교가 됩니다. 저희는 **총 학습 FLOPs 매칭 + 추론 토큰/연산 동등**(PaSST는 학습 시 Patchout, **추론은 풀토큰**)으로 **비용을 고정**한 채 신뢰성을 향상시켰습니다.   (Green AI 취지·지표화 논의는 CACM를 참고해 주세요. ([Communications of the ACM][8]))

### 2‑C. “얼마나 안정적으로” — 다중 분할·시드·시나리오에서 **일관성**

* **LODO(거리 시나리오 보류)에서도 개선 유지**, 위험@커버리지(예: risk@80%) 감소가 재현됩니다. 
* **편집 셋의 안정성(시드·체크포인트·투영 간 Jaccard/Spearman 일치)**을 보고, 과도한 편향/왜곡이 없는지도 점검했습니다. 

> **정리:** **동일 추론비용** 아래 **신뢰성 지표 개선**이 **여러 분할/시나리오에서 통계적으로 유의**하고, **편집 규모가 작으며 재현성 자료(해시·스크립트)까지 공개**되었다면, SOTA 정확도를 ‘이기지 않아도’ 학술적으로 **충분한 의미**가 있습니다. (GRSL 원고는 분할 매니페스트/지표 스크립트 공개를 명시) 

---

## 2.1) 예: 최고 90%, 단순 네트워크 80%에서 **데이터 조리로 82%** → 의미 있나?

**가능합니다. 다만 아래 조건을 충족할 때 ‘의미 있음’으로 설득력이 커집니다.**

1. **통계적 유의성**: 단순 독립 근사로 95% 신뢰구간 폭을 ±1% 이하로 하려면 (n \gtrsim 0.8\cdot0.2\cdot(1.96/0.01)^2 \approx 6{,}147) 테스트 표본이 필요합니다(정확도 80% 가정). 실제론 **쌍대 비교**(같은 샘플에 대한 두 분류기) 구조이므로 **McNemar/부트스트랩**으로 **짝지어진 오차 감소**가 유의한지 검정하는 게 정석입니다.
2. **신뢰성 동반 개선**: 80→82% **정확도 + ECE↓ + AURC/AUGRC↓**를 **동일 추론비용**에서 보이면, **알람 임계치 안정·오탐 감소** 같은 운영 이득이 명확합니다. (ICASS P 원고는 “보정 안정화→알람 임계치 안정”을 실무적 가치로 언급) 
3. **일관성/외삽성**: **다양한 데이터셋/분할/시드**에서 **동일한 향상 경향**이 재현되어야 합니다(저희 LODO·다중 시드 보고). 
4. **공정비교**: **학습 FLOPs·추론비용 통제**(Compute‑Fairness, Inference Parity)를 명시하세요. 

> **즉, 82%가 90% SOTA를 못 이겨도**—**값비싼 모델을 쓰지 않고(추론비용 동일)** **신뢰성/일관성**을 동반 개선한다면 **충분히 가치 있는 기여**입니다. 이는 **Selective Prediction**(Risk–Coverage) 문헌의 목표와도 부합합니다. ([NeurIPS Proceedings][7])

---

## 3) **경량화된 인공지능**—제한된 온보드 연산(무인체계 등)에서의 실전 레시피

아래는 **“구조는 고정, 데이터/미세조정/양자화로 배포 최적화”**를 전제로 한 **우선순위형 레시피**입니다.

### 3‑A. 백본·미세조정

* **강한 사전학습 백본 + 얕은 헤드**: AST/PaSST 등 사전학습 오디오 트랜스포머를 **동결**하고 헤드만 학습하면, 저장·추론 경량성과 **영향도 계산의 안정성**이 좋아집니다. (ICASS P/GRSL 모두 이 정책을 채택)
* **PEFT(파라미터 효율 미세조정)**: **Adapters/LoRA/SSF**. 특히 **SSF는 스케일·시프트 재매개변수를 **추론 시 합쳐(merge)** **추가 추론비용 0**을 달성합니다. (GRSL) 
* **훈련‑전용 효율화**: PaSST의 **Patchout**은 학습만 가속하고 **추론은 풀토큰**으로 맞춰 **비용 공정성**을 지킵니다. (원 논문 권고) ([ISCA Archive][9])

### 3‑B. 양자화(모바일/엣지 배포)

* **TensorFlow Lite Post‑Training Quantization(PTQ)**: **INT8** 정수 양자화로 **추론 지연·전력·모델 크기**를 줄입니다(대표 데이터 몇 샘플로 보정). ([TensorFlow][10])
* 필요 시 **Quantization‑Aware Training(QAT)**로 정확도 저하를 추가 완화할 수 있습니다. (일반적으로 TFLite INT8은 모델 크기 ~4× 축소, **1.5–4× 추론 가속**의 보고) ([TensorFlow][11])
* 구현 가이드(공식): PTQ / QAT 참조. ([TensorFlow][10])

### 3‑C. 데이터 중심 확보(경량화와 상보)

* **검증 전용 영향도 기반 큐레이션**으로 **≤1% 소량 편집**만으로 **신뢰성 지표 개선**을 먼저 달성하면, **추가적인 모델 경량화(양자화·PEFT)에도 성능 유지가 용이**합니다. (GRSL: ID/LODO에서 AURC/AUGRC·ECE 일관 개선) 
* **액티브 셀렉션**은 **양자화 전‑후 공통의 ‘라벨 예산 효율’** 향상을 도와, **소형 온보드 모델**의 데이터 요구량 자체를 줄입니다. (ICASS P: Neg‑Influence/Entropy 결합이 예산‑선정률 우수)

---

## (보충) 본 초고들에서의 근거와 최근 문헌의 정합성

* **우리 결과**: VTUAD에서 **편집 ≤1%**만으로 **AUGRC↓·AURC↓·ECE↓**를 **동일 추론비용** 하에 일관되게 달성(LODO 포함), risk@80%도 **≈9–12% 상대감소**. 데이터 편집은 **시드·체크포인트·투영**에 **안정적**.
* **방법론적 정합성**: 영향도(Influence Functions/TracIn/TRAK), 라벨 노이즈 정제(Cleanlab), 부분집합 선택(GLISTER), 선택적 예측(Risk–Coverage)이 **데이터 중심** 주제의 표준 축을 형성. ([Proceedings of Machine Learning Research][12])
* **공정비교의 필요**: **Green AI**가 제안한 효율/정확도 동시 보고 프레임과 부합(학습 FLOPs·추론동등성 통제). ([Communications of the ACM][8])

---

## 실무용 체크리스트 (요약)

1. **지표 세트**: Accuracy + **AURC(보정 전)**, **AUGRC(보정 후)**, **ECE/NLL/Brier** — 운영 임계치 안정성까지 보고. 
2. **비용 통제**: 총 학습 FLOPs 매칭, **추론 토큰/연산 동등성**(PaSST는 학습 Patchout, 추론 풀토큰). 
3. **통계/재현성**: 시드 다중화·LODO/ID 재현, 부트스트랩 CI/짝지은 검정, **편집 셋 해시·스크립트 공개**. 
4. **경량화 배포**: PEFT(특히 **SSF Merge**) → TFLite **INT8 PTQ/QAT** 순으로 진행, 정확도/신뢰성 점검표 유지.  ([TensorFlow][10])

---

### 참고한 초고(첨부)

* **ICASS P 제출본**: *Influence‑Aware Curation and Active Selection for Industrial and Surveillance Sound Events* — **동결 백본 + 헤드 전용 TracIn**, 소규모 큐레이션/선정으로 정확도·보정 동시 개선. 
* **GRSL 투고본**: *Reliability‑First Passive Underwater Acoustics via Influence‑Guided Curation* — **VTUAD**에서 **Compute‑Fairness & Inference Parity** 하 **AURC/AUGRC·ECE** 일관 개선(≤1% 편집). 

### 핵심 외부 근거(발췌)

* NeurIPS **Datasets & Benchmarks** 트랙 개설 및 가이드라인. ([NeurIPS][1])
* Andrew Ng의 **Data‑Centric AI** 정의와 동향. ([IEEE Spectrum][2])
* **TRAK/TracIn/Influence Functions** (데이터 기여도/영향도). ([Proceedings of Machine Learning Research][3])
* **Cleanlab/Confident Learning** (라벨 노이즈 정제). ([Google Research][4])
* **GLISTER** (부분집합 선택). ([dblp.org][5])
* **Selective Prediction**(Risk–Coverage, SelectiveNet). ([NeurIPS Proceedings][7])
* **Green AI** (효율·정확도 동시 평가). ([Communications of the ACM][8])
* **AST / PaSST** (음향 트랜스포머; PaSST Patchout는 학습 전용). ([ISCA Archive][6])
* **TFLite PTQ/QAT** (INT8 양자화, 성능/전력/크기 절감). ([TensorFlow][10])

---

필요하시면, 위 체크리스트에 맞춘 **실험 표준운영절차(SOP) 초안**과 **통계 검정/CI 산출 스크립트(부트스트랩 + McNemar)**, **TFLite INT8 변환 파이프라인**을 정리해서 같이 드리겠습니다.

[1]: https://neurips.cc/Conferences/2020/DatasetsBenchmarks/AcceptedPapers?utm_source=chatgpt.com "Datasets and Benchmarks Accepted Papers"
[2]: https://spectrum.ieee.org/andrew-ng-data-centric-ai?utm_source=chatgpt.com "Andrew Ng: Unbiggen AI - IEEE Spectrum"
[3]: https://proceedings.mlr.press/v202/park23c.html?utm_source=chatgpt.com "TRAK: Attributing Model Behavior at Scale"
[4]: https://research.google/pubs/confident-learning-estimating-uncertainty-in-dataset-labels/?utm_source=chatgpt.com "Confident Learning: Estimating Uncertainty in Dataset Labels"
[5]: https://dblp.org/rec/conf/aaai/KillamsettySRI21.html?utm_source=chatgpt.com "dblp: GLISTER: Generalization based Data Subset Selection for Efficient and Robust Learning."
[6]: https://www.isca-archive.org/interspeech_2021/gong21b_interspeech.html?utm_source=chatgpt.com "ISCA Archive - AST: Audio Spectrogram Transformer"
[7]: https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html?utm_source=chatgpt.com "Selective Classification for Deep Neural Networks"
[8]: https://cacm.acm.org/magazines/2020/12/248800-green-ai/fulltext?utm_source=chatgpt.com "Green AI – Communications of the ACM"
[9]: https://www.isca-archive.org/interspeech_2022/koutini22_interspeech.html?utm_source=chatgpt.com "ISCA Archive - Efficient Training of Audio Transformers with Patchout"
[10]: https://www.tensorflow.org/model_optimization/guide/quantization/post_training?utm_source=chatgpt.com "Post-training quantization  |  TensorFlow Model Optimization"
[11]: https://www.tensorflow.org/model_optimization/guide/quantization/training?utm_source=chatgpt.com "Quantization aware training  |  TensorFlow Model Optimization"
[12]: https://proceedings.mlr.press/v70/koh17a?utm_source=chatgpt.com "Understanding Black-box Predictions via Influence Functions"
