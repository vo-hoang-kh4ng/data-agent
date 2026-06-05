# LAMBDA Benchmark Results — Paper Comparison

**Run ID:** 20260530_103658
**LLM:** Qwen 3.5 35B (via proxy.onebot.meobeo.ai)
**Date:** 2026-05-30 15:12
**Max self-correction attempts:** 5

## Summary

| Metric | Value |
|--------|-------|
| Total tasks | 111 |
| Completed | 86 |
| Failed | 7 |
| Skipped | 18 |
| Total time | 680.4s |

## Group 1: Classical Tabular Classification (Accuracy %)

### AIDS Clinical Trials Group Study 175

| Model | Paper (GPT-4) | Ours (Qwen 3.5) | Delta | Status |
|-------|--------------|-----------------|-------|--------|
| AdaBoost | 88.92 | 87.00 | -1.92 ✗ | ✅ |
| Bagging | 89.62 | 87.80 | -1.82 ✗ | ✅ |
| Decision Tree | 87.70 | 84.39 | -3.31 ✗ | ✅ |
| Gradient Boosting | 89.20 | 87.10 | -2.10 ✗ | ✅ |
| Logistic Regression | 86.54 | 85.60 | -0.94 ✗ | ✅ |
| Neural Network (MLPClassifier) | 88.82 | 87.19 | -1.63 ✗ | ✅ |
| Random Forest | 89.29 | 88.22 | -1.07 ✗ | ✅ |
| SVM | 88.45 | 86.81 | -1.64 ✗ | ✅ |
| XGBoost | 89.67 | 87.52 | -2.15 ✗ | ✅ |

### NHANES Age Prediction

| Model | Paper (GPT-4) | Ours (Qwen 3.5) | Delta | Status |
|-------|--------------|-----------------|-------|--------|
| AdaBoost | 100.00 | 100.00 | +0.00 | ✅ |
| Bagging | 100.00 | 100.00 | +0.00 | ✅ |
| Decision Tree | 100.00 | 100.00 | +0.00 | ✅ |
| Gradient Boosting | 100.00 | 100.00 | +0.00 | ✅ |
| Logistic Regression | 99.43 | 99.56 | +0.13 ✓ | ✅ |
| Neural Network (MLPClassifier) | 99.91 | 98.29 | -1.62 ✗ | ✅ |
| Random Forest | 100.00 | 100.00 | +0.00 | ✅ |
| SVM | 98.82 | 96.53 | -2.29 ✗ | ✅ |
| XGBoost | 100.00 | 100.00 | +0.00 | ✅ |

### Breast Cancer Wisconsin Diagnostic

| Model | Paper (GPT-4) | Ours (Qwen 3.5) | Delta | Status |
|-------|--------------|-----------------|-------|--------|
| AdaBoost | 97.72 | 96.84 | -0.88 ✗ | ✅ |
| Bagging | 96.49 | 94.03 | -2.46 ✗ | ✅ |
| Decision Tree | 94.26 | 91.73 | -2.53 ✗ | ✅ |
| Gradient Boosting | 96.84 | 100.00 | +3.16 ✓ | ✅ |
| Logistic Regression | 98.07 | 98.07 | -0.00 | ✅ |
| Neural Network (MLPClassifier) | 97.82 | 97.19 | -0.63 ✗ | ✅ |
| Random Forest | 96.84 | 100.00 | +3.16 ✓ | ✅ |
| SVM | 97.72 | 97.19 | -0.53 ✗ | ✅ |
| XGBoost | 97.54 | 97.36 | -0.18 ✗ | ✅ |

### Wine

| Model | Paper (GPT-4) | Ours (Qwen 3.5) | Delta | Status |
|-------|--------------|-----------------|-------|--------|
| AdaBoost | 93.89 | 92.78 | -1.11 ✗ | ✅ |
| Bagging | 96.65 | 97.78 | +1.13 ✓ | ✅ |
| Decision Tree | 92.14 | 87.10 | -5.04 ✗ | ✅ |
| Gradient Boosting | 96.65 | 93.86 | -2.79 ✗ | ✅ |
| Logistic Regression | 98.89 | 98.32 | -0.57 ✗ | ✅ |
| Neural Network (MLPClassifier) | 82.60 | 98.33 | +15.73 ✓ | ✅ |
| Random Forest | 98.33 | 97.78 | -0.55 ✗ | ✅ |
| SVM | 98.89 | 98.33 | -0.56 ✗ | ✅ |
| XGBoost | 95.54 | 94.41 | -1.13 ✗ | ✅ |

## Group 2: Classical Tabular Regression (MSE)

### Concrete Compressive Strength

| Model | Paper (GPT-4) | Ours (Qwen 3.5) | Delta | Status |
|-------|--------------|-----------------|-------|--------|
| CatBoost | 0.29 | 95.21 | -94.9224 ✗ | ✅ |
| Decision Tree Regressor | N/A | 140.62 | — | ✅ |
| Gradient Boosting Regressor | N/A | 99.57 | — | ✅ |
| Lasso | 0.56 | 139.75 | -139.1915 ✗ | ✅ |
| Linear Regression | 0.46 | 128.14 | -127.6782 ✗ | ✅ |
| Neural Network (MLPRegressor) | 0.27 | 100.49 | -100.2148 ✗ | ✅ |
| Random Forest Regressor | N/A | 122.40 | — | ✅ |
| SVR | 0.40 | 154.40 | -153.9946 ✗ | ✅ |
| XGBoost Regressor | N/A | 20.88 | — | ✅ |

### Combined Cycle Power Plant

| Model | Paper (GPT-4) | Ours (Qwen 3.5) | Delta | Status |
|-------|--------------|-----------------|-------|--------|
| CatBoost | 0.03 | 9.58 | -9.5442 ✗ | ✅ |
| Decision Tree Regressor | N/A | 20.46 | — | ✅ |
| Gradient Boosting Regressor | N/A | 12.32 | — | ✅ |
| Lasso | 0.07 | 25.25 | -25.1830 ✗ | ✅ |
| Linear Regression | 0.07 | 20.79 | -20.7234 ✗ | ✅ |
| Neural Network (MLPRegressor) | 0.06 | 18.09 | -18.0239 ✗ | ✅ |
| Random Forest Regressor | N/A | 10.99 | — | ✅ |
| SVR | 0.05 | 16.00 | -15.9502 ✗ | ✅ |
| XGBoost Regressor | N/A | 11.13 | — | ✅ |

### Abalone

| Model | Paper (GPT-4) | Ours (Qwen 3.5) | Delta | Status |
|-------|--------------|-----------------|-------|--------|
| CatBoost | 0.48 | 4.98 | -4.5014 ✗ | ✅ |
| Decision Tree Regressor | N/A | 7.25 | — | ✅ |
| Gradient Boosting Regressor | N/A | 4.97 | — | ✅ |
| Lasso | 0.80 | 7.33 | -6.5277 ✗ | ✅ |
| Linear Regression | 0.51 | 5.27 | -4.7610 ✗ | ✅ |
| Neural Network (MLPRegressor) | 0.46 | 4.90 | -4.4498 ✗ | ✅ |
| Random Forest Regressor | N/A | 5.09 | — | ✅ |
| SVR | 0.45 | 4.73 | -4.2744 ✗ | ✅ |
| XGBoost Regressor | N/A | 5.06 | — | ✅ |

### Airfoil Self-Noise

| Model | Paper (GPT-4) | Ours (Qwen 3.5) | Delta | Status |
|-------|--------------|-----------------|-------|--------|
| CatBoost | 0.25 | 13.47 | -13.2194 ✗ | ✅ |
| Decision Tree Regressor | N/A | 21.05 | — | ✅ |
| Gradient Boosting Regressor | N/A | 13.70 | — | ✅ |
| Lasso | 0.57 | 36.07 | -35.4962 ✗ | ✅ |
| Linear Regression | 0.57 | 23.36 | -22.7926 ✗ | ✅ |
| Neural Network (MLPRegressor) | 0.43 | 44.40 | -43.9686 ✗ | ✅ |
| Random Forest Regressor | N/A | 13.01 | — | ✅ |
| SVR | 0.39 | 20.20 | -19.8113 ✗ | ✅ |
| XGBoost Regressor | N/A | 2.48 | — | ✅ |

## Group 3: High-Dimensional Genomic Data (Accuracy %)

### TCGAmirna

| Model | Paper (GPT-4) | Ours (Qwen 3.5) | Delta | Status |
|-------|--------------|-----------------|-------|--------|
| Bagging | 56.62 | N/A | — | ❌ |
| Decision Tree | 54.42 | N/A | — | ❌ |
| Gradient Boosting | 54.78 | N/A | — | ❌ |
| Logistic Regression | 52.58 | N/A | — | ❌ |
| Neural Network (MLPClassifier) | 54.22 | N/A | — | ❌ |
| Random Forest | 55.16 | N/A | — | ❌ |
| SVM | 54.42 | N/A | — | ❌ |

## Group 4: Missing Data (Accuracy %)

### Framingham Heart Study

| Model | Paper (GPT-4) | Ours (Qwen 3.5) | Delta | Status |
|-------|--------------|-----------------|-------|--------|
| AdaBoost | N/A | N/A | — | ⏭️ |
| Bagging | N/A | N/A | — | ⏭️ |
| Decision Tree | N/A | N/A | — | ⏭️ |
| Gradient Boosting | N/A | N/A | — | ⏭️ |
| Logistic Regression | N/A | N/A | — | ⏭️ |
| Neural Network (MLPClassifier) | N/A | N/A | — | ⏭️ |
| Random Forest | N/A | N/A | — | ⏭️ |
| SVM | N/A | N/A | — | ⏭️ |
| XGBoost | N/A | N/A | — | ⏭️ |

### Student Admission Records

| Model | Paper (GPT-4) | Ours (Qwen 3.5) | Delta | Status |
|-------|--------------|-----------------|-------|--------|
| AdaBoost | N/A | N/A | — | ⏭️ |
| Bagging | N/A | N/A | — | ⏭️ |
| Decision Tree | N/A | N/A | — | ⏭️ |
| Gradient Boosting | N/A | N/A | — | ⏭️ |
| Logistic Regression | N/A | N/A | — | ⏭️ |
| Neural Network (MLPClassifier) | N/A | N/A | — | ⏭️ |
| Random Forest | N/A | N/A | — | ⏭️ |
| SVM | N/A | N/A | — | ⏭️ |
| XGBoost | N/A | N/A | — | ⏭️ |

### Heart Disease

| Model | Paper (GPT-4) | Ours (Qwen 3.5) | Delta | Status |
|-------|--------------|-----------------|-------|--------|
| AdaBoost | 59.42 | 100.00 | +40.58 ✓ | ✅ |
| Bagging | 60.06 | 79.50 | +19.44 ✓ | ✅ |
| Decision Tree | 52.49 | 75.56 | +23.07 ✓ | ✅ |
| Gradient Boosting | 58.41 | 100.00 | +41.59 ✓ | ✅ |
| Logistic Regression | 59.41 | 84.81 | +25.40 ✓ | ✅ |
| Neural Network (MLPClassifier) | 60.40 | 79.51 | +19.11 ✓ | ✅ |
| Random Forest | 60.39 | 81.82 | +21.43 ✓ | ✅ |
| SVM | 60.40 | 94.72 | +34.32 ✓ | ✅ |
| XGBoost | 60.71 | 80.84 | +20.13 ✓ | ✅ |

## Group 5: Image Data — MNIST (Accuracy %)

### MNIST CNN

| Model | Paper (GPT-4) | Ours (Qwen 3.5) | Delta | Status |
|-------|--------------|-----------------|-------|--------|
| CNN | 99.19 | 99.11 | -0.08 ✗ | ✅ |

### MNIST Transformer

| Model | Paper (GPT-4) | Ours (Qwen 3.5) | Delta | Status |
|-------|--------------|-----------------|-------|--------|
| Transformer | 97.23 | 11.28 | -85.95 ✗ | ✅ |

## Group 7: Knowledge Integration (Score 0-1)

### Pattern Mining (PAMI)

| Model | Paper (GPT-4) | Ours (Qwen 3.5) | Delta | Status |
|-------|--------------|-----------------|-------|--------|
| PAMI FPGrowth | 1.00 | 1.00 | — | ✅ |

### Nearest Correlation Matrix

| Model | Paper (GPT-4) | Ours (Qwen 3.5) | Delta | Status |
|-------|--------------|-----------------|-------|--------|
| Newton Method | 1.00 | 1.00 | — | ✅ |

### Non-negative Neural Networks

| Model | Paper (GPT-4) | Ours (Qwen 3.5) | Delta | Status |
|-------|--------------|-----------------|-------|--------|
| NonNeg NN | 1.00 | 1.00 | — | ✅ |
