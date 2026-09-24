# Lab 2 — KNN and Radius Neighbors on the Wine Dataset

## Purpose

This lab compares two distance-based classifiers on the sklearn Wine dataset (178 wines, 13 chemical measurements, 3 cultivars). K-Nearest Neighbors votes among a fixed *count* of nearest training points; Radius Neighbors votes among every point inside a fixed *distance*. The aim is to see how each parameter shapes accuracy and to work out when one design is preferable to the other.

KNN was run at k = 1, 5, 11, 15, 21 and RNN at radius = 350, 400, 450, 500, 550, 600, on an 80/20 stratified split.

## Files

| File | Description |
|------|-------------|
| `knn_rnn_analysis.ipynb` | Full notebook — all four lab steps with code, plots, and output |

## Key insights

**KNN flattens out immediately.** Accuracy is 0.778 at k = 1 and 0.806 at every larger k tested. Only k = 1 underperforms, which is the textbook signature of overfitting — a single neighbour lets one atypical wine decide a prediction. Past k = 5 the additional votes change no decisions at all.

**RNN gets steadily worse as the radius grows**, falling from 0.722 at radius 350 to 0.667 at 600. The reason shows up in the neighbourhood-size diagnostic: at radius 350 an average test point already contains 80 of the 142 training points, and at 600 it contains 111. These are not local neighbourhoods — the model is averaging over most of the dataset, so predictions drift toward the overall class distribution.

**The real story is feature scaling.** Proline carries **99.77%** of the total variance in the unscaled data, because it is measured in the hundreds while most other features sit between 0 and 5. Unscaled Euclidean distance is therefore very nearly a one-dimensional measurement of proline, and twelve of the thirteen features are effectively discarded. That is what caps both models in the 0.7–0.8 range. Standardising the features lifts KNN from about 0.68 to about 0.97 under 5-fold cross-validation.

**KNN beat RNN at every single setting**, by about 11 percentage points on average. The advantage is structural: guaranteeing exactly k votes makes KNN indifferent to local density, and k = 5 means the same thing on any dataset. A radius does not transfer — it has to be re-tuned whenever the scaling or feature set changes.

## Challenges and decisions

**RNN crashes on isolated points.** `RadiusNeighborsClassifier` raises `ValueError` if a test point has no neighbour inside the radius. Setting `outlier_label='most_frequent'` makes it fall back to the majority class instead. At the prescribed radii this never actually fires — the diagnostic confirms zero empty neighbourhoods — but it is required for the smaller radii used in the scaling experiment.

**The prescribed radii only make sense on unscaled data.** Radii of 350–600 look absurd until you realise they are effectively proline units. Rather than quietly substituting scaled features, the lab was run exactly as specified and the scaling investigation added afterwards as a separate section, so the prescribed results stand on their own.

**After standardisation the radius grid had to be rebuilt from scratch.** Every scaled point lies within roughly 10 units of every other, so 350–600 would admit the entire dataset. A grid of 2–8 was needed instead. This is itself the clearest evidence that a radius is not a portable parameter: at radius 8 the scaled model lands exactly on the 0.389 majority-class baseline, fully degenerate.

**The test set is small.** 36 samples means one prediction is worth 2.78 percentage points, so small gaps between settings are noise and the occasional perfect score is better read as "indistinguishable from very good". 5-fold cross-validation was added to confirm the headline scaled/unscaled gap holds across folds rather than resting on one lucky split.

## Running it

```bash
pip install pandas numpy matplotlib scikit-learn
jupyter notebook knn_rnn_analysis.ipynb
```

The dataset ships with scikit-learn, so no download is needed. Run all cells top to bottom; the split uses a fixed random seed, so results reproduce exactly.
