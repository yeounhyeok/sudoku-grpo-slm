# 학습 설계 (GRPO Zero, v3 → v4)

## 방식
- 시작 모델: 원본 Qwen3.5-2B(IT), thinking ON. SFT 없음. 풀이 전략은 지정하지 않고 형식만 제약(`<think>` → `<answer>` 9줄 격자).
- 보상: 정답(유효 격자 + 정답 일치) + 형식. 판정은 `src/sudoku_grpo/reward.py`(평가와 같은 규칙, 다중해 없음 확인).
- 알고리즘: TRL `GRPOTrainer` 서브클래스(`scripts/train_grpo.py`), 그룹 G=8, 스텝당 8문제, lr 5e-6, β=0.005, token-level loss(TRL `dapo`), 토큰 절단 IS, 생성 상한 24,576토큰.
- rollout 디코딩은 평가와 맞춤: T=0.6, top-p 0.95, top-k 512, presence penalty 1.5.

## v3 → v4
- v3: 난이도 혼합 풀. 그룹의 약 37%가 전부 정답 또는 전부 오답이라 advantage가 0이었다.
- v4: medium 전용 풀(`rl_mpool`, 320문제) + **online dynamic sampling**(DAPO 요소 중 이것만 차용): 8개 전부 맞거나 전부 틀린 그룹은 버리고 같은 스텝에서 풀에서 다시 뽑는다(`--dynamic-sampling --ds-max-rounds 3`, `merge_groups`, 단위 테스트 `scripts/test_merge_groups.py`).
- clip-higher(ε_high)는 `num_iterations=1`이라 ratio=1이어서 효과가 없어 제거했다. 길이 보상은 넣지 않았다(대조군 유지).

## 이어 돌리기(차수)
디스크(`/root` 74GB) 한계로 한 번에 끝내지 않고 차수로 나눠 돌리고, 차수마다 직전 병합 모델에서 시작한다(옵티마이저는 초기화).

| 차수 | 시작 모델 | 시드 | 누적 스텝 | 스크립트 |
|---|---|---|---|---|
| 1 | v3 step60 병합본 | 6 | 1~30 | `run_grpo_zero_v4.sh` |
| 2 | v4 step30 병합본 | 7 | 31~60 | `run_grpo_zero_v4_c2.sh` |
| 3 | v4 step60 병합본 | 8 | 61~100 (진행 중) | `run_grpo_zero_v4_c3.sh` |

병합: `scripts/merge_lora.py <adapter> <out> --base <직전 병합본>`.

## 중단 기준
KL > 0.01, entropy < 0.08, fmt_wrong 연속 5% 초과, 잘림률 지속 상승, 체크포인트 test 평가가 v3s60보다 명확히 하락.

## 하드웨어
- 학습: eney(MI250, ROCm, FP8 없음). 커널은 `scripts/setup_kernels.sh`.
- 평가: 한성대 공용 GPU 서버 1장.
