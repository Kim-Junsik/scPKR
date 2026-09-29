# scPKR — single-cell Pathway Koopman Residual

조합 유전자·약물 섭동에 대한 단일세포 발현 예측. 학습에서 본 적 없는 조합의 반응을,
그 조합을 이루는 단일들만 보고 예측한다.

이름의 세 조각이 곧 방법이다. **P**athway — KEGG 소속 행렬이 Koopman **관측 함수**다
(학습하지 않는다). **K**oopman — 그 고정 관측 공간에서 섭동마다 선형 연산자로 전개한다.
**R**esidual — 모델은 가법 성분을 만들지 않는다. 가법 성분은 닫힌 형태 ridge에서 오고,
모델은 **비가법 잔차만** 담당한다.

```
Δx(S, x) = Σ_{a∈S} w_a                        ← ridge, 유전자 공간, 신호의 88%
         + 1[|S| ≥ 2] · W · B_S · Φ(x)        ← 경로 뼈대, 비가법 12%

Φ(x) = [M x ; S x]                 M = KEGG 소속 행렬(행 정규화), S = 앵커 유전자. 고정
A_a  ∈ R^{K×K}                     섭동별 Koopman 연산자, 공유 기저 + 저계수
B_S  = Σ_{a<b∈S} (A_a A_b + A_b A_a)          반교환자
W    ∈ R^{G×K}, KEGG 엣지에서만 비영
```

`A_a = 0` 초기화에서 **예측이 정확히 `ridge_additive`** 이고, `|S| = 1`이면 둘째 항이
비어 **단일 조건은 구조적으로 ridge 그대로**다. 즉 학습은 측정된 출발점에서 내려가는
방향으로만 움직인다.

설계의 근거와 검증 계획은 [docs/DESIGN.md](docs/DESIGN.md)에 있다.

---

## 넘어야 할 선 — 측정됨, 추측 아님

`scripts/baseline_l2.py`, 5 fold, 보고된 프로토콜(테스트 부분집합 scanpy HVG 1,000개,
soft gate). Control L2가 문헌 값과 0.2~2.5% 안에서 일치하므로 축척이 검증되어 있다.

| | ridge_additive | scDFM | scPKFM(v1) |
|---|---|---|---|
| Table 1 Norman additive | **1.669** | 1.704 | 1.957 |
| Table 2 Single | **1.416** | 1.619 | 1.754 |
| Table 2 Double | 2.251 | **2.031** | 2.375 |
| Table 3 ComboSciPlex | 1.858 | **1.657** | 2.138 |

**`ridge_additive`가 이 저장소의 출발점이고 동시에 상대다.** 상호작용 항도 신경망도 flow도
없는 닫힌 형태 ridge이며, 결정론적이라 seed 잡음이 없다. 그리고 v1(scPKFM)을 네 표 모두에서
이긴다.

학습 모델이 ridge를 이기는 것은 **"단일이 홀드아웃된 조합 블록"에서만**이고, 두 번 모두
정확히 0.20이다(Table 2 Double +0.220, Table 3 +0.201). 단일이 전부 학습에 있으면(Table 1)
이점이 사라지고, **홀드아웃 단일에서는 학습 모델이 0.20 손해를 본다** — scDFM도 같이.

**따라서 이 저장소의 과제는 하나다: 조합 블록에서 비가법 0.20.** 단일 블록은 건드리지
않는다.

목표선을 직접 확인:

```bash
python scripts/baseline_l2.py --set data.cache_h5ad=assets/combosciplex_scanpy5000_additive_scdfm7_fold0.h5ad \
    data.raw_h5ad=data/combosciplex/combosciplex.h5ad data.control_label=control \
    data.n_hvg=5000 data.hvg_criterion=scanpy data.normalise_from_counts=counts \
    split.source=list split.fold=0 --group all --fallback control
```

`control` 행이 5.3606으로 나와야 한다. 그게 프로토콜이 보고된 표와 같은지에 대한 검증이고,
다르면 다른 어떤 행도 인용하면 안 된다.

---

## v1(scPKFM)에서 무엇을 가져왔고 무엇을 버렸는가

v1은 `tables-v1` 태그로 동결되어 있다. 보고된 표, 체크포인트, 12 run의 ablation 이력이
거기 있고 그것이 논문 부록의 증거다. 지우지 않는다.

**가져온 것** — 아키텍처와 무관하고 다시 만들면 3주다.

| | 왜 |
|---|---|
| `src/data/` | scDFM와 동일한 분할, 누수 차단(held-out 조건을 HVG 선택에서까지 배제) |
| `src/data/preprocess.py` | combosciplex counts → 중앙값 라이브러리 재정규화. **이거 하나로 Control L2가 5.33 vs 8.25** |
| `src/eval/baselines.py` | `ridge_additive` = 출발점이자 상대 |
| `src/eval/conditions.py` | 채점 조건과 유전자 공간. v1의 diagnostics에서 모델 의존성을 뗀 것 |
| `src/eval/scdfm_metrics.py` | + cell-eval 0.5.42 / pdex 0.1.28 핀. 이 핀을 찾는 데 든 시간이 코드보다 크다 |
| `src/models/heads.py` | hurdle. 데이터의 41%가 정확히 0이고 `soft → sample`이 edist를 6.64 → 1.46 |
| `scripts/dev_queue.py`, `dev_score.py` | 계측기. 3 seed, fold·seed 내 짝지은 차이, 사전 등록 채택 규칙 |
| `scripts/baseline_l2.py` | 보고 지표로 베이스라인을 재는 유일한 도구 |

**버린 것** — 측정이 닫은 방향이다.

| | 증거 |
|---|---|
| 잠재 오토인코더 (P-CAB / E-RCA) | Table 3에서 디코더 한계가 0.97 = 전체 오차의 45%. 수송 전에 지불되고 제거 불가 |
| 인코더 용량 확대 | rank-16 readout: 단일 블록 2.57 → 3.94 |
| KL 상향 | 1e-2에서 Pearson Δ 0.012 — 잠재 붕괴 |
| ρ (학습된 합성 보정) | 홀드아웃 조합 3개 중 2개를 해쳤다(+0.40, +0.31). 단독으로는 단일을 3.53까지 악화 |
| Lie 괄호 상호작용 항 | 7개 설정·2개 백본에서 5전 5패. **틀린 대칭류를 겨냥했기 때문** (아래) |
| operator graph (구조 유사도 결합) | penalty·mix 둘 다 개선 없음 |

---

## 세 가지 검증 가능한 주장

각각 빼면 무엇이 무너지는지가 정의되어 있다. `docs/DESIGN.md` §3에 ablation 설계가 있다.

**A. KEGG가 원칙적 Koopman 관측 사전이다.**
eDMD의 중심 미해결 문제가 "관측 함수 사전을 어떻게 고르는가"이고, 기존 답은 다항식·RBF·
학습된 인코더였다. KEGG는 생물학에서 오므로 과적합하지 않고, 고정이므로 재구성 손실이 없다.
Ablation: 같은 차원의 무작위 뼈대, PCA 사전, 상위 분산 유전자.

**B. 비가법성은 경로를 통해 매개된다.**
가법 효과는 유전자 공간 전체에서 자유롭지만 비가법 보정은 KEGG 엣지 위에서만 발생한다.
생물학적 주장이고 반증 가능하다. Ablation: `W`를 dense로.

**C. 동시 섭동 합성의 올바른 대칭은 반교환자다.**

```
exp(τ(A+B)) − (exp(τA) + exp(τB) − I) = (τ²/2)(AB + BA) + O(τ³)
```

교환자 `[A,B]`는 반대칭이므로 **순서 의존성**을 잡는 항이고, 동시 이중 섭동은 교환에
대칭이라 감독받을 신호가 없다. **이것이 v1의 Lie 괄호 항이 5전 5패한 이유이고, 이 문헌의
상호작용 항들이 대체로 같은 실수를 한다.** 수치 확인은 DESIGN.md §2.2에 있다.

---

## 설치

**1. KEGG 경로 주석.** 저장소에 포함하지 않는다(재배포가 별개 허가). `assets/kegg/`가
이미 있으면 건너뛴다.

```bash
python scripts/download_kegg.py
```

**2. 데이터.** 용량 때문에 포함하지 않는다.

```
data/norman/norman.h5ad                 Norman et al. 조합 섭동 scRNA-seq
data/norman/split_results.pkl           5-fold 분할 (데이터와 함께 배포)
data/combosciplex/combosciplex.h5ad     ComboSciPlex (정규화된 X + raw counts 층)
```

**3. 캐시.**

```bash
python data_prepare.py --set data.n_hvg=5000 data.hvg_criterion=scanpy
```

`assets/*.h5ad`는 fold별로 만들어진다. `data/`와 `assets/*.h5ad`는 `.gitignore`에 있으므로
git으로 옮겨지지 않는다.

---

## 현재 상태

인프라와 목표선만 있다. **모델은 아직 없다.** 다음 순서로 만든다:

1. `src/config.py`의 `model` 절 — 관측 함수, 연산자 크기, `W` 희소도
2. `src/models/observables.py` — `Φ`와 KEGG 행렬, `W`의 희소 구조
3. `src/models/operator.py` — `A_a`, `B_S`, `matrix_exp` 흐름 맵
4. `tests/test_structure.py` — 초기화 시점 예측 = ridge, 단일 = ridge, 순열 불변,
   쌍으로 인덱싱되는 파라미터 없음

4번이 이 저장소의 핵심 불변식이다. 학습 없이 성립해야 하므로, 실패하면 "학습이 부족하다"가
아니라 **구성이 틀렸다**는 뜻이다.
