# DAY 2 모델 개발 및 평가

1. **문제:** 초기 100사이클 → 제공 cycle_life 회귀. X: delta_logvar, policy_C1, policy_switch_SOC_pct, policy_C2.
2. **분할:** Batch 1 학습 46개 / Batch 2 테스트 39개. Batch 3 미사용.
3. **최종 선정:** 세 후보의 Batch 2 MAPE 최소 → **Random Forest**. 개발 CV 우승 모델은 Ridge.
   Batch 2를 선정에 사용했으므로 별도의 독립 최종 테스트는 아직 없음.
   CV MAE 81.33 ± 23.77 cycles (fold 표준편차).
   최종 파라미터는 Hold-out 제외 개발 구간 그룹5-fold에서 결정: {'regressor__model__max_depth': 5, 'regressor__model__min_samples_leaf': 2}.
   Pipeline 안에서 fold별 전처리, ln/exp 타깃 변환. 파라미터 튜닝·fit은 Batch 1만 사용; 최종 모델 종류는 Batch 2 MAPE로 선정.
4. **Batch 2:** MAE **148.36 cycles**, RMSE **157.70 cycles**,
   R² **0.483**, MAPE **30.18%**.
   39개 중 35개 과대 예측;
   평균 편향(예측−실제) +129.79 cycles.
5. **배치 차이:** 신규 명목 Policy 29/39개;
   delta_logvar 학습 범위 밖 11개. 평균 수명 Batch 1 844.7 →
   Batch 2 565.7 cycles. 특정 조건이 오차의 주원인이라고 단정하지 않음.
6. **라벨 민감도:** 점검 후보10개 제외, Hold-out을 뺀 개발 28개에서 선정하고
   전체 Batch 1 36개로 같은 파라미터를 재학습한
   Random Forest, 같은 Batch 2 MAPE 29.59%. 주 실험 선정 유지.
7. **논문 참고 목표:** 초록 9.1% 대비 **+21.08%p**, 미달.
   MAPE = 100 × mean(abs(actual−predicted)/actual).
   [논문 초록](https://www.nature.com/articles/s41560-019-0356-8), [Table 1·Methods](https://web.mit.edu/braatzgroup/Severson_NatureEnergy_2019.pdf).
   논문 pooled train41/primary43/secondary40과 본 과제 B1→B2는 다르고 B2 파일 날짜도 다름.
   동일 조건 재현 또는 논문 모델 대비 우열의 증거로 해석하지 않음.
   Table 1 primary 값은 이상 셀 포함; 괄호의 한 셀 제외값과 구분함.
   Full 모델은 온도 센서 문제로 별도 제외한 셀도 있음. 평가 오차를 보고 셀을 제외하지 않음.
   논문 EOL 정의는 정격1.1Ah의80%(0.88Ah); 본 실험은 제공 cycle_life 라벨 사용.
8. **최종 모델:** final_model.joblib은 Batch 1 46개로만 fit. 평가 후 Batch 2/3 재학습 없음.
   이전 EDA·검증의 Batch 2 노출 이력이 있어 신규 블라인드 테스트는 아님.

## 검증·테스트 요약

                   stage  MAPE_or_gap unit
      Train (Batch 1 CV)       9.7266    %
Valid (Batch 1 Hold-out)       8.4435    %
          Test (Batch 2)      30.1792    %
       Gap (Train-Valid)      -1.2831   %p
        Gap (Valid-Test)      21.7358   %p
       Gap (Target-Test)      21.0792   %p

Hold-out은 개발 모델로 평가하고, Batch 2는 같은 파라미터로 전체 Batch 1 재학습한 모델로 평가한다.

## Batch 2 점수

experiment         model  selected_before_test  train_batch  test_batch  test_n      MAE     RMSE      R2  MAPE_pct  selected_final
   main_46         Ridge                  True            1           2      39 228.9143 257.4030 -0.3773   48.7767           False
   main_46 Random Forest                 False            1           2      39 148.3631 157.7025  0.4830   30.1792            True
   main_46       XGBoost                 False            1           2      39 166.5420 184.7256  0.2907   33.6489           False

## 논문 참고 성능

  source             split     model     RMSE  MAPE_pct  our_MAPE_minus_reference_pp
Abstract     대표 test error  Headline      NaN    9.1000                      21.0792
 Table 1 Primary (이상 셀 포함)  Variance 138.0000   14.7000                      15.4792
 Table 1 Primary (이상 셀 포함) Discharge  91.0000   13.0000                      17.1792
 Table 1 Primary (이상 셀 포함)      Full 118.0000   14.1000                      16.0792
 Table 1         Secondary  Variance 196.0000   11.4000                      18.7792
 Table 1         Secondary Discharge 173.0000    8.6000                      21.5792
 Table 1         Secondary      Full 214.0000   10.7000                      19.4792

## 다음 실험

모델 개선은 Batch 1 내부에서 사전에 계획한 방식으로 검증하고 새로운 평가 배치에서 확인한다.
제공 수명 라벨의 종단 완결성도 점검할 필요가 있다.

