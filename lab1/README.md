# Lab 1 — Data Visualization, Preprocessing, and Statistical Analysis


## What's covered

**Visualization** — scatter plot of petal length vs. width, box plots of every feature by species, and a histogram of sepal length.

**Preprocessing** — mean imputation for missing values, IQR-based outlier detection, reduction via 70% sampling plus dropping `sepal width`, then Min-Max scaling, Z-score standardization, and binning petal length into categories.

**Statistics** — central tendency and dispersion measures for each feature, plus a correlation matrix and heatmap.

## Notes on the results

The IQR method flagged zero outliers in petal length, which fits the data: these are natural botanical measurements, not readings prone to sensor error. Missing values were introduced synthetically (15 total, fixed seed) since the source dataset is complete — the point was to demonstrate imputation, not to repair real gaps.

Petal measurements separate the three species far more cleanly than sepal measurements do, which is why `sepal width` was the column dropped during dimension elimination. Petal length and petal width correlate at r = 0.93, strong enough that feeding both into a predictive model would introduce multicollinearity.

## Running it

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
jupyter notebook data_analysis.ipynb
```

Run all cells top to bottom; later cells depend on DataFrames built earlier. Sampling uses a fixed random seed, so results reproduce exactly.
