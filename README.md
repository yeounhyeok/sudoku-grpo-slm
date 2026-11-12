# sudoku-grpo-slm

한성대 대형언어 모델 기말 프로젝트 (2026-2) — **Qwen3.5-2B가 정해진 24k 토큰 예산 안에서 스도쿠를 얼마나 푸는가**를 GRPO(R1-Zero 방식) 강화학습으로 높이고, Base 대비 성능을 정량 비교한다.

- 초기 방법 참고(현재는 폐기): [G-RaR: From SFT to GRPO for Structured Reasoning (Kaggle)](https://www.kaggle.com/code/srishanya/g-rar-from-sft-to-grpo-for-structured-reasoning/notebook) — 이 노트북은 JAX/TPU 기반이라 코드를 가져오지 않고 방법론만 PyTorch(TRL)로 재구현한다.
- 공고 요건: Base ≤ 4B(공개 모델), 실제 파인튜닝, Base vs FT 정량 비교, Train/Test 분리
- 일정: 중간발표자료 2026-11-23 00:00 / 최종 2026-12-15 00:00

## 상태 (2026-11-12)
Qwen3.5-2B를 **SFT 없이 GRPO(R1-Zero 방식)** 로 학습한다. 초기에 시도한 SFT→GRPO(G-RaR)는 SFT 모델이 학습 전보다 나빠져(21.9% → 0~6.3%) 폐기했다.

| 단계 | 상태 |
|---|---|
| 모델 선정·데이터 준비·채점 유틸 | ✅ `configs/`, `scripts/prepare_data.py`, `src/sudoku_grpo/` |
| Baseline 평가 | ✅ [docs/baseline.md](docs/baseline.md) — 공식 base(thinking ON·32k) test_id 21.9% |
| SFT (G-RaR 경로) | ❌ 폐기 (악화) — 기록은 Obsidian `_legacy` |
| GRPO Zero v3 (v3s60) | ✅ 원본 돌파: 전체 24k 30.9% (base 18.8%) |
| GRPO Zero v4 (online dynamic sampling, medium 풀) | 🔄 누적 60스텝 완료, 60→100스텝 3차 진행 중 |
| SFT 데이터 생성(rejection sampling) | ⬜ 100스텝 이후 검토 |
| HRM 체크포인트 비교 | ⬜ 미착수 |

결과 상세·재현 설정: **[docs/results.md](docs/results.md)**, 학습 설계: **[docs/training.md](docs/training.md)**.

### 핵심 수치 (test_id 64문제, k=8, T0.6, presence 1.5, 생성 상한 32k, 정답률 %)
| 모델 | 전체 24k | 전체 32k | medium 24k | hard 32k |
|---|---|---|---|---|
| base (Qwen3.5-2B IT) | 18.8 | 21.5 | 21.3 | 1.4 |
| v3 step60 | 30.9 | 38.5 | 46.9 | 7.4 |
| v4 step60 | 37.7 | 45.3 | 62.5 | 11.6 |

표본이 64문제(medium 24개)라 표준오차가 크다(medium 24k ±7%p 안팎). 같은 64문제로 반복 평가했으므로 선택 편향도 있다.

## 시작(Base) 모델 (HF revision 고정, post-trained/IT 체크포인트)
| key | repo | 파라미터(safetensors) | 라이선스 |
|---|---|---|---|
| `qwen3.5-0.8b` | [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | 0.87B | Apache-2.0 |
| `qwen3.5-2b` | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | 2.27B | Apache-2.0 |

선택: 같은 계열 0.8B/2B로 모델 크기 효과를 비교한다. `-Base`(사전학습 전용) 체크포인트는 `--all`로 받을 수 있다.
Gemma-4-E2B는 전체 파라미터가 5.12B라 "4B 이하" 요건에 위배될 수 있어 제외했다.
Qwen3.5는 Vision-Language 구조(`Qwen3_5ForConditionalGeneration`)다. 텍스트 전용 학습 시 TRL/PEFT와의 호환은 **아직 미검증**.

## 데이터
| split | 원본 | 크기 | 용도 |
|---|---|---|---|
| `sft_train` | beta3/GridCorpus (cc-by-4.0) | 18,800 | SFT |
| `rl_train` | 동일 | 3,700 | GRPO |
| `val` | 동일 | 370 | 하이퍼파라미터/체크포인트 선택 |
| `test_id` | 동일 | 760 | 최종 평가(분포 내) |
| `test_ood` | sapientinc/sudoku-extreme `test.csv` | 1,000 | 최종 평가(어려운 분포 밖, rating 구간 4×250) |

- 난이도 층화: beta3는 `difficulty_tier`(easy/medium/hard/expert), sapient는 tdoku backtrack rating 구간.
- **split 간 정답 그리드(solution) 겹침 없음**(스크립트가 검사, 실행 시 중복 발견하면 중단). sapient는 공식 test만 사용(train.csv 미사용).
- 시드 고정·결정적: 같은 설정이면 `data/manifest.json`의 sha256이 동일하다(2회 실행으로 확인).
- beta3 CSV는 5.7GB라 앞부분만 스트리밍한다(약 7만 행, 수십 초). 앞부분 편향 가능성은 완전히 배제하지 못함.
- 데이터 자체는 커밋하지 않는다(`data/processed/`는 .gitignore). 재생성: `make data`.
- 퍼즐 단서 수: beta3 평균 약 37칸, sapient OOD 평균 약 25칸(훨씬 어려움).
- 후보 비교는 [docs/datasets.md](docs/datasets.md).

## 사용법
```bash
bash scripts/setup_env.sh            # BACKEND=rocm|cu128|cpu (기본 rocm 7.2)
source .venv/bin/activate
python scripts/download_models.py    # models/ (git 제외), 0.8B+2B 약 6.3GB
python scripts/prepare_data.py       # data/processed/*.jsonl + data/manifest.json
python tests/test_grid.py
```
환경 변수 `MODELS_DIR`로 모델 저장 위치를 바꿀 수 있다. eney 파드에서는 저장소·모델·venv를 영속 볼륨인 `/root` 아래에 둘 것.

## 레코드 / 프롬프트 형식
- 퍼즐·정답은 81자 문자열(행 우선, 빈칸 `.`). jsonl: `id, puzzle, solution, source, tier|rating, givens`
- 모델 출력: `<reasoning>…</reasoning><answer>9줄 그리드</answer>` (`src/sudoku_grpo/prompt.py`, 파서/채점은 `grid.py`)

## 구조
```
configs/    models.toml, data.toml
scripts/    download_models.py, prepare_data.py, setup_env.sh, train_grpo.py(TRL GRPOTrainer 확장: dynamic sampling),
            run_grpo_zero_v{3,4,4_c2,4_c3}.sh, eval_k.py, merge_lora.py
src/sudoku_grpo/   grid.py(검증·파싱·정확도), prompt.py
tests/      test_grid.py
data/       manifest.json (processed/는 git 제외)
docs/       datasets.md, baseline.md, results.md, training.md
```

## 데이터 출처·라이선스
- beta3/GridCorpus_9M_Sudoku_Puzzles_Enriched — CC BY 4.0
- sapientinc/sudoku-extreme — 라이선스 태그 없음(데이터 카드의 사용 지침: train만 학습, test로 평가)
- 코드: MIT (`LICENSE`)

## 학습 커널 (Qwen3.5 GDN)

`scripts/setup_kernels.sh` — flash-linear-attention 0.6.0 + causal-conv1d 1.7.0(HIP 빌드, C++20 패치). MI250 2B 학습 처리량 LoRA ×1.76, Full FT ×1.97, 수치 동일(`results/bench/train_kernels_2b.jsonl`, `scripts/bench_train.py`).
