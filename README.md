# sudoku-grpo-slm

한성대 대형언어 모델 기말 프로젝트 (2026-2). G-RaR 방식(SFT → GRPO)으로 소형 LM이 스도쿠를 잘 풀도록 파인튜닝.

- 참고: https://www.kaggle.com/code/srishanya/g-rar-from-sft-to-grpo-for-structured-reasoning/notebook
- 공고 요건: Base 4B 이하, 실제 FT, Base vs FT 정량 비교, Train/Test 분리
- 마감: 중간발표자료 2026-11-23 00:00 / 최종 2026-12-15 00:00
- 컴퓨팅: eney (MI250 64GB, ROCm 7.2, 파드 RAM 64GiB/CPU 12)

## 후보 Base 모델 (HF 조회 2026-11-04)
| 모델 | safetensors 파라미터 수 |
|---|---|
| Qwen/Qwen3.5-0.8B (+ -Base) | 0.87B |
| Qwen/Qwen3.5-2B (+ -Base) | 2.27B |
| google/gemma-4-E2B (+ -it) | 5.12B (⚠️ 4B 초과 가능, 공고 해석 확인 필요) |

데이터 후보는 `docs/datasets.md` 참고.
