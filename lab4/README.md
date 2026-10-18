# Lab 4 — Regression Techniques: Linear, Multiple, Polynomial, Ridge, and Lasso

## Purpose

This lab explores five regression approaches on the sklearn Diabetes dataset (442 samples, 10 physiological measurements predicting one-year disease progression). The goal is to understand when each method excels, how regularization prevents overfitting, and what tradeoffs exist between predictive power and model simplicity. All models are evaluated on the same 80/20 train-test split using MAE, MSE, RMSE, and R².

## Files

| File | Description |
|------|-------------|
| `regression_analysis.ipynb` | Full notebook — all six steps with code, results, visualizations |

## Key insights

**Multiple regression dominates.** Using all ten features yields R² = 0.4526, far superior to simple linear regression on BMI alone (R² = 0.2334). Disease progression is genuinely multi-dimensional — no single measurement suffices.

**Polynomial regression adds no value here.** Degree-2 polynomial on BMI achieved R² = 0.2290, worse than linear. The relationship is approximately linear; fitting a curve overfits the noise in a small test set.

**Regularization trades training accuracy for stability.** Ridge (α=1.0) dropped R² from 0.4526 to 0.4192; Lasso (α=0.5) stayed at 0.4441. The slight regression in training fit is offset by improved generalization — regularized models are more likely to perform consistently on fresh data.

**Lasso performs automatic feature selection.** While Ridge retained all 10 coefficients (just scaled down), Lasso zeroed out 6 features, identifying 4 as truly informative. This sparsity is interpretable: a clinician looking at the model can see which measurements matter, whereas Ridge's 10 non-zero coefficients require explaining their relative magnitudes.

**Regularization strength matters.** An alpha-sweep showed both Ridge and Lasso peak around R² 0.45–0.47. Too little regularization (alpha ≈ 0) overfits; too much (alpha > 100) underfits by shrinking all coefficients to near-zero. The "sweet spot" depends on the dataset.

**Ridge vs Lasso in practice:**
- Ridge: better when you believe all features contribute, even weakly; good for prediction
- Lasso: better when you want interpretability; assumes most features are noise

## Challenges and decisions

**Standardized data arrived ready.** sklearn's Diabetes dataset is pre-standardized (mean ≈ 0, std ≈ 1). This is ideal for regularized models, where the penalty term scales with coefficient magnitude — large-magnitude features would dominate an unscaled problem. No additional scaling was needed.

**Small test set causes noise.** With only 89 test samples, one mispredict changes R² by ~1%. The R² = 0.2290 for polynomial regression might reflect overfitting or might reflect the smaller hypothesis space (one feature) — the gap is real but jittery. Cross-validation would smooth this, but the single split was sufficient for the lab's purpose.

**Lasso alpha tuning was exploratory.** The default alpha=1.0 showed good performance, but an alpha-sweep revealed the landscape: R² peaks around 0.5 and degrades smoothly as alpha grows. Rather than grid-search, I picked α=0.5 after inspection, trading a small R² gain (0.4441 vs 0.4526) for interpretability (four features instead of ten).

**Visualizations emphasized comparison.** Six charts (scatter plots, actual-vs-predicted, coefficient bars, performance metrics, and alpha sweep) make the comparisons concrete. Tables alone would not communicate the regularization effect or the multi-dimensionality of the data.

## Running it

```bash
pip install numpy pandas matplotlib scikit-learn seaborn
jupyter notebook regression_analysis.ipynb
```

The Diabetes dataset ships with scikit-learn, so nothing needs downloading. Run all cells top to bottom; the random seed (random_state=42) ensures reproducibility.


