# Baseline (시작 모델, 학습 전) — 2026-11-04

> 용어: "Base" = 파인튜닝 전 시작 모델. 사용 모델은 `Qwen/Qwen3.5-0.8B`·`Qwen3.5-2B` **post-trained(IT)** 체크포인트이며, 순수 사전학습 `-Base`는 사용하지 않았다.

## 🔒 평가 프로토콜 (고정, 11/4) — 학습 전·후 동일 적용

- 명령: `scripts/eval.py --mode think --n 64 --seed 0` (split: test_id, test_ood). 시작 모델과 SFT·GRPO 체크포인트 모두 **이 명령 그대로**
- 같은 문제: `random.Random(0)`로 고른 같은 64문제 (seed 고정이라 매번 동일)
- 같은 프롬프트·템플릿: `SYSTEM_THINK` + chat template thinking ON, 채점은 `</think>` 뒤 `<answer>`만
- 같은 디코딩: T=1.0, top_p=0.95, top_k=20, min_p=0, presence_penalty=1.5, 상한 32,768, 생성 seed 0, 1샘플(pass@1)
- 같은 채점: exact(헤드라인) / valid / format / cell_acc + 토큰 수·잘림·think 종료율
- **학습 설계 제약**: SFT trace는 `<think>` 블록 안에 넣고 최종 격자는 `<answer>`로 → 학습 후 모델도 같은 thinking ON 경로로 평가 가능
- 보고: 정답률과 함께 **평균 생성 토큰**도 비교 (같은 32k 예산에서 얼마나 짧게 푸는지)
- 한계: n=64는 신뢰구간이 넓음(대략 ±10%p대, 추정). 최종 발표에서 n을 키우면 시작 모델도 같은 n으로 재측정

## 1차 — greedy, thinking OFF

환경: eney (MI250X, torch 2.14.1+rocm7.2), HF `generate`, bf16, **greedy**, thinking 모드 OFF(`enable_thinking=False`).
Qwen3.5의 가속 커널(flash-linear-attention, causal_conv1d)은 미설치라 PyTorch 대체 경로로 돌았다(정확도엔 영향 없음, 속도만 느림).
채점: `exact`(81칸 완전 일치) / `valid`(행·열·박스 충돌 없음 + 단서 보존) / `format`(`<answer>`에 81칸) / `cell_acc`(원래 빈칸 중 맞은 비율).
원시 요약 JSON: `results/baseline/`. 생성 샘플은 용량 때문에 커밋하지 않았다(`outputs/`, git 제외).

## 1. Direct (추론 없이 격자만 출력, 최대 400토큰, n=200)
| 모델 | split | exact | valid | format | cell_acc | 평균 출력 토큰 |
|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | test_id | 0.000 | 0.000 | 0.990 | 0.114 | 166 |
| Qwen3.5-0.8B | test_ood | 0.000 | 0.000 | 0.970 | 0.106 | 167 |
| Qwen3.5-2B | test_id | 0.000 | 0.000 | 1.000 | 0.131 | 165 |
| Qwen3.5-2B | test_ood | 0.000 | 0.000 | 1.000 | 0.108 | 169 |

- `cell_acc ≈ 0.11`은 **1~9를 무작위로 찍을 때의 기대값(1/9 = 0.111)** 과 같다 → 직접 출력 baseline은 사실상 무작위 수준.
- 난이도(tier / rating 구간)별로 의미 있는 차이 없음(0.10~0.15, n이 작아 노이즈).
- 모델 크기 0.8B→2B 효과는 test_id에서 +0.017(칸 정확도)로 미미.

## 2. Reason (`<reasoning>` → `<answer>` 프롬프트, 최대 4096토큰, test_id 파일럿 n=32)
| 모델 | exact | format | 잘림(truncated) | 평균 출력 토큰 |
|---|---|---|---|---|
| Qwen3.5-0.8B | 0/32 | 0.00 | 100% | 4096 |
| Qwen3.5-2B | 1/32 (easy 1) | 0.03 | 97% | 4074 |

- 0.8B: `<reasoning>`/`<answer>` 태그를 한 번도 쓰지 않고 "Step 1: Analyze Row 1 …"식으로 길게 풀다가 "Wait, 중복이다…" 자기점검을 반복, 결론에 도달하지 못함.
  반복 루프는 아님(마지막 200줄 중 최다 반복 줄 4~23회, 출력 6.5k~8.5k자).
- 2B: 32개 중 31개가 4096토큰에서 잘림. 1개만 3401토큰에 풀이 완료(easy).
- 1024토큰(8문제 스모크)에서도 0.8B는 8/8 잘림. 상한을 늘려도 "답까지 가지 못하는" 양상이 바뀌지 않음.
- **n=32 파일럿이라 신뢰구간이 넓다**(1/32 = 3.1%는 통계적으로 0과 구분 어려움). 본 평가 수치로 쓰지 말 것.

## 해석 (추정 포함)
- 사실(실측): 두 모델 모두 Base 상태에서 스도쿠 완전 정답률은 ≈0%. direct 모드는 무작위 수준, reason 모드는 길이 상한 안에 결론을 못 낸다.
- 해석(추정): 병목은 (a) 형식 불이행, (b) 추론이 수렴하지 않음, (c) 토큰 효율(숫자·공백이 토큰을 많이 소비해 격자를 다시 쓸 때마다 예산이 소모됨). SFT는 (a)와 (c)를, GRPO는 (b)를 겨냥하는 것이 자연스럽다.
- 한계: greedy 단일 시드, 샘플링(temperature) 평가 미실시. thinking 모드 ON은 미평가 → 2차에서 측정. pass@k 미측정.
- 4096은 우리가 정한 생성 상한이다(모델 `max_position_embeddings`=262,144).
- 학습 후 비교는 같은 split·같은 채점·같은 greedy 설정으로 해야 공정하다(`scripts/eval.py`).

## 재현
```bash
python scripts/eval.py --model qwen3.5-0.8b --split test_id --n 200 --batch-size 50 --mode direct
python scripts/eval.py --model qwen3.5-2b   --split test_id --n 32  --mode reason --tag _pilot
```
(`--n`은 시드 0으로 부분 샘플링. 전체 split을 쓰려면 `--n 0`.)


## 2차 (공식) — thinking ON, Qwen3.5 모델카드 sampling, 32k, vLLM

- 명령: `/opt/vllm-venv/bin/python scripts/eval.py --model <M> --mode think --backend vllm --eager --no-prefix-cache --split test_id,test_ood --n 64`
- 엔진: vLLM 0.33.0+rocm72, eager (MI250/gfx90a에서 HIP graph는 FULL·PIECEWISE 모두 `HSA_STATUS_ERROR_INVALID_PACKET_FORMAT`로 크래시)
- 디코딩: T=1.0, top_p=0.95, top_k=20, min_p=0, presence_penalty=1.5, max 32,768, seed 0, 1 sample
- 원시 결과: `results/baseline/*-think-qwen-think-vllm_*.json`

| 모델 | split | exact | 잘림(32k) | 평균 토큰 |
|---|---|---|---|---|
| 2B | test_id | **14/64 (21.9%)** | 78% | 30,159 |
| 2B | test_ood | 0/64 | 100% | 32,768 |
| 0.8B | test_id | 0/64 | 100% | 32,768 |
| 0.8B | test_ood | 0/64 | 100% | 32,768 |

- 2B test_id 난이도별: easy 7/8 · medium 7/24 · hard 0/27 · expert 0/5
- 2B: thinking을 끝낸 14개는 **전부 정답**(끝냈는데 오답 0). 정답 trace 길이 7.7k~29k 토큰
- 토큰 예산별 정답률(2B test_id, 같은 런에서 사후 계산): ≤4k 0% · ≤8k 1.6% · ≤16k 4.7% · ≤32k 21.9%
- 0.8B는 64/64 모두 32k에서 thinking 중 잘림
- 소요: 2B 2,751초, 0.8B 2,636초 (vLLM eager, 각 128문제 동시)
- 해석: 1차의 ≈0%는 상당 부분 **4096 상한 탓**이었음(2B 한정). 파인튜닝 목표 = ① hard·expert·OOD 정답률 ② 토큰 수 2~3만 → ~1k로 축소

이 설정이 학습 전·후 비교의 고정 프로토콜이다(위 🔒 절 참고, 엔진도 동일하게 유지).
