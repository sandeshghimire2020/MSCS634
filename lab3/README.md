# Lab 3 — K-Means and K-Medoids Clustering on the Wine Dataset

## Purpose

This lab compares two partitional clustering algorithms on the sklearn Wine dataset (178 wines, 13 chemical measurements, 3 cultivars). Both split the data into k groups by distance, but they differ in what a cluster centre is allowed to be: K-Means uses a **centroid**, the arithmetic mean of its members, which can sit anywhere; K-Medoids uses a **medoid**, an actual observation chosen to minimise total distance to the rest of its cluster.

Both were fitted with k = 3 on z-score standardised features and scored on Silhouette (geometric quality, no labels used) and Adjusted Rand Index (agreement with the true cultivars).


## Key insights

**K-Means won on both metrics.**

| | Silhouette | ARI | Misgrouped |
|---|---|---|---|
| K-Means | 0.2849 | 0.8975 | 6 / 178 |
| K-Medoids | 0.2676 | 0.7411 | 16 / 178 |

The ARI gap is the meaningful one — K-Means misgroups 6 wines against 16, so K-Medoids makes roughly two and a half times as many errors.

**The two methods disagree in one specific place.** They agree closely overall (ARI 0.77 between the two partitions) and both recover two of the three cultivars *perfectly*. All the divergence sits in the third cultivar, the one positioned between the other two in chemical space. K-Medoids pulls a block of those middle wines into a neighbouring cluster. The cause is the medoid constraint: a centroid can settle in the sparse gap between groups, which is exactly where a dividing centre is most useful, while a medoid is pinned to the location of a real wine. That restriction buys robustness to outliers — and this dataset has none worth worrying about, so here it only costs accuracy.

**A low silhouette score does not mean a bad clustering.** Both scores sit below 0.30 even though K-Means reaches ARI 0.90. The metrics answer different questions: silhouette asks whether clusters are compact and separated, ARI asks whether they are correct. The cultivars genuinely adjoin one another rather than forming isolated balls, so a near-perfect recovery still scores modestly on geometry.

**The most important result is a trap.** Clustering the *unscaled* data gives a much **better** silhouette (0.5711 vs 0.2849) but a much **worse** ARI (0.3711 vs 0.8975). Unscaled, one feature — proline, measured in the hundreds while most others sit between 0 and 5 — dominates the distance calculation, so K-Means carves tidy bands along that single axis. Those bands are geometrically crisp and factually wrong. Since silhouette consults no labels, on a genuine unlabelled problem it would have endorsed the wrong answer with confidence. Internal metrics reward tidiness, not truth.

To be fair to silhouette: on *standardised* data it does correctly peak at k = 3. The failure came from the feature scaling, not the metric.

## Challenges and decisions

**sklearn has no K-Medoids, so PAM was implemented by hand.** The usual third-party source, `scikit-learn-extra`, was last released in 2022 and does not work against the installed sklearn 1.7. Rather than pin an abandoned dependency or risk downgrading a working environment, Partitioning Around Medoids is written out directly in the notebook: a BUILD phase that greedily selects k starting medoids, then a SWAP phase that exchanges medoids with non-medoids while total cost falls. It uses only numpy and scipy, keeps the notebook self-contained for a grader, and arguably satisfies "implement the K-Medoids algorithm" better than an import would.

**The scatter plots required a projection, and it is lossy.** The data has 13 dimensions, so the side-by-side plots use PCA to reach 2D. That view retains only **55.4%** of total variance, meaning about 45% of the structure is invisible on the page. Points that look overlapping may be well separated along a discarded axis, so the plots are a sketch rather than a map. The notebook states this rather than presenting the projection as the full picture.

**Marking the two centre types needed different handling,** which turned out to be instructive. K-Means centroids are synthetic coordinates and had to be passed through the same PCA transform to be plotted. Medoids are real rows, so plotting them is just an index lookup — which is the conceptual difference between the algorithms made directly visible.

**K-Means is mildly seed-dependent.** ARI ranges from 0.8975 to 0.9149 across ten random seeds, so `random_state=42` and `n_init=10` are fixed throughout for reproducibility. The reported figures are from one specific seed and would wobble slightly on another.

## Running it

```bash
pip install pandas numpy matplotlib scikit-learn scipy
jupyter notebook clustering_analysis.ipynb
```

The dataset ships with scikit-learn and K-Medoids is implemented in the notebook, so there is nothing to download and no extra packages beyond the above. Run all cells top to bottom.
