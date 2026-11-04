# 스도쿠 데이터셋 후보 (HF API 조회 2026-11-04, 라이선스는 태그 기준)

| 데이터셋 | 행 수 | 라이선스 | 비고 |
|---|---|---|---|
| sapientinc/sudoku-extreme | 4,254,780 | 태그 없음 | 난이도(rating) 있음, 어려운 퍼즐 위주(HRM 계열) |
| abatilo/sudokubench | 1,292,800 | apache-2.0 | puzzle/solution 81자, 시작 칸 수 있음 |
| certen/Sudoku-1M-Puzzles | 1,000,000 | mit | puzzle/solution만, 난이도 없음 |
| nakashi104/sudoku-1million | 1,000,000 | cc0-1.0 | Kaggle 1M 계열, steps(풀이 단계) 포함 |
| Ritvik19/Sudoku-Dataset | 17,001,130 | apache-2.0 | 난이도/풀이시간 포함, 2.2GB(원본 기준) |
| beta3/GridCorpus_9M_Sudoku_Puzzles_Enriched | 7,520,090 | cc-by-4.0 | 난이도 tier·naked/hidden single 수 등 메타데이터 풍부 |
| twistshan/Sudoku_DLM_Reasoning | 210,752 | other | 난이도 버킷·분할(train/OOD) 이미 정의됨 |
| Guoen/sudoku_cot | 18,030 | 태그 없음 | CoT 포함이나 중국어, 백트래킹 trace |
| qgallouedec/sudoku | 500 | 태그 없음 | 난이도 라벨(very_easy 등), 평가용 소량 |
| SakanaAI/sudoku-bench-nikoli | 100 | cc-by-4.0 | 변형 규칙 퍼즐, 이번 과제 범위 밖 |

미확인: 각 데이터의 중복·train/test 누수, 풀이(reasoning) 품질.

## 공개 벤치마크와의 관계
- `sapientinc/sudoku-extreme`는 HRM(arXiv:2506.21734)·TRM(arXiv:2510.04871)이 쓰는 Sudoku-Extreme 벤치마크다(두 공식 레포의 `build_sudoku_dataset.py`가 이 데이터셋의 `test.csv` 전체로 평가).
- 우리 test_ood는 그 `test.csv`의 rating 층화 부분집합(1,000)이며 평가는 그중 64문제 → 논문 수치와 직접 비교하지 않는다.
- Sudoku-Bench(Sakana AI, arXiv:2505.16135)는 변형 규칙 퍼즐 중심이라 범위 밖(확장 후보).

## 채택 (2026-11-04)
- 학습/검증/ID 테스트: beta3/GridCorpus (난이도 tier·메타데이터, cc-by-4.0)
- OOD 테스트: sapientinc/sudoku-extreme `test.csv` (rating 구간 층화)
- 풀이 trace(SFT 정답 추론)는 공개 데이터가 마땅치 않아 솔버로 직접 생성할 예정 (미구현)
