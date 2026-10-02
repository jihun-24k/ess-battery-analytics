# 데이터 안내

원본 MATLAB v7.3(HDF5) 파일은 `data/archive/`에 둡니다. `src/preprocess.py`는 필요한 필드만 h5py로 읽습니다. 원본 파일은 수정하지 않습니다.

| Batch | 파일 | 원본 셀 | 유효 수명 라벨 | 기본 FE 표본 |
|---|---|---:|---:|---:|
| 1 | `2017-05-12_batchdata_updated_struct_errorcorrect.mat` | 46 | 46 | 46 |
| 2 | `2018-02-20_batchdata_updated_struct_errorcorrect.mat` | 47 | 39 | 39 |
| 3 | `2018-04-12_batchdata_updated_struct_errorcorrect.mat` | 46 | 44 | 40 |

`2018-04-03_varcharge_batchdata_updated_struct_errorcorrect.mat`는 이번 세 배치 분석 대상에 포함하지 않습니다.

원본 필드 중 `summary.cycle`은 실제 사이클 번호, `QDischarge`는 방전 용량(Ah), `cycles.Qdlin`은 공통 전압 격자의 방전곡선입니다. 충전 조건은 `policy_readable`에서 읽습니다. `cycle_life`는 제공된 최종 수명 라벨입니다. 결측 수명은 대체하지 않고 제외합니다. 셀 ID는 원본의 0부터 시작하는 인덱스를 유지하며, `sample_id=b{batch}_c{cell}`로 세 배치를 구분합니다.

기본 Feature Engineering에서는 Batch 3의 원본 품질 플래그 셀 `{2,37,42,43}`을 추가 제외합니다. Batch 1 종단 점검 후보 10개는 주 실험에 유지하고, 이들을 제외한 36개로 별도 라벨 민감도 분석을 합니다. 종단 용량, 최종 사이클, ID, 배치 번호는 모델 입력에 넣지 않습니다.

## 생성 데이터

`02_feature_engineering.ipynb` 또는 다음 명령으로 `data/processed/`를 생성합니다.

```bash
.venv/bin/python -m src.features
```

- `X.csv`: 125행 × 4개 입력 피처
- `y.csv`: 동일한 `sample_id`의 `cycle_life`
- `metadata.csv`: 배치·Policy·품질·라벨 점검 정보
- `sensitivity/`: Batch 1 점검 후보를 추가 제외한 115행
- 피처 대안, 정의, 결측·상관·VIF 점검 표, `manifest.json`

초기 피처는 실제 10~100사이클에서만 계산합니다. `delta_logvar=log10(var(Q100(V)-Q10(V)))`이며 분산은 `ddof=0`입니다. 기본 X는 `delta_logvar`, `policy_C1`, `policy_switch_SOC_pct`, `policy_C2`입니다.

## 논문과의 관계

참고 연구는 [Severson et al. (2019)](https://www.nature.com/articles/s41560-019-0356-8)입니다. 본 프로젝트 Batch 2 파일 날짜는 논문의 두 번째 실험 날짜와 다르고, 수명 라벨 정리·학습/평가 분할·입력 피처도 다릅니다. 논문의 성능은 참고값이며 이 데이터 구성으로 동일 조건 재현을 주장하지 않습니다.
