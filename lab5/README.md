# Lab 5 — Hierarchical and DBSCAN Clustering on the Wine Dataset

## Purpose

This lab applies two clustering methods to the sklearn Wine dataset (178 wines, 13 standardized chemical measurements, 3 cultivars). Agglomerative hierarchical clustering builds a tree of merges and cuts it at a chosen number of clusters. DBSCAN finds dense regions and labels sparse points as noise. The aim is to see how each method's parameters shape the result, to score both against the known cultivars, and to judge which suits this kind of data.

## Files

| File | Description |
|------|-------------|
| `hierarchical_dbscan_analysis.ipynb` | Full notebook covering all four steps, with code, output, and five charts |

## Key insights

**Ward agglomerative clustering with three clusters recovered the cultivars best.** It reached homogeneity 0.790 and completeness 0.783, with a silhouette of 0.277, and produced groups of 58, 56, and 64 wines. Silhouette peaked at three clusters too, which agrees with the three known classes.

**Linkage choice matters as much as the cluster count.** Ward, complete, average, and single linkage were compared at k = 3. Average and single linkage chained the data together and left 174 wines in one cluster, with homogeneity near 0.01 to 0.02. Complete linkage was workable (homogeneity 0.62) but weaker than Ward.

**The dendrogram shows a clear structure.** The merges below the top are packed closely together. The last two merges jump sharply, from about 12.6 to 27.7 to 35.4, which is a natural place to stop and suggests a small number of groups.

**DBSCAN was highly sensitive to its parameters on the full data.** Across the grid of eps and min_samples, most settings either merged everything into one cluster or labelled a large share of wines as noise. The best homogeneity was 0.50, at eps 2.5 with min_samples 10, which produced only two clusters and 32 noise points.

**Reducing to three principal components helped DBSCAN.** On PCA(3), which keeps 66.5% of the variance, eps 1.0 with min_samples 10 produced three clusters, with homogeneity 0.703 and completeness 0.554, while labelling 48 wines as noise. This is a real improvement, but it still trails Ward, and the projection discards about a third of the variance.

**Overall, Ward clustering was the stronger choice for this dataset.** The cultivars form compact groups in moderate dimension, which suits a method that merges by variance. DBSCAN's density-based approach needs clearer density differences than these data provide.

## Challenges and decisions

**Choosing eps was hard.** The k-distance plot is supposed to show a knee that suggests a good eps. In 13 dimensions the curve climbs gradually with no sharp elbow, because distances between points bunch together. I used the plot to narrow the search to roughly 2 to 3, then tested a grid rather than trusting a single value.

**Silhouette is not comparable across the two methods.** DBSCAN's silhouette score is computed only on non-noise points, which are the easiest to place, so it looks much higher (0.571) than Ward's (0.277), which includes every wine. For the comparison I relied on homogeneity and completeness, which treat noise as its own label and are therefore fairer. The notebook states this caveat directly.

**Homogeneity rises with the number of clusters almost by construction.** More clusters means purer clusters, so choosing k by homogeneity alone would always favour the largest value. Completeness, which falls as clusters are added, is what keeps the choice honest. I printed both and noted this in the notebook.

**Dimensionality reduction was a deliberate trade-off.** Projecting to three components improved DBSCAN but threw away a third of the variance. I treated the PCA result as a separate experiment and kept the full-dimensional results alongside it, so the cost of the projection is visible rather than hidden.

**Dendrogram readability.** A full dendrogram over 178 wines is unreadable, so I truncated it to the last 20 merges and marked the 2- and 3-cluster cut heights on the plot.

## Running it

```bash
pip install numpy pandas matplotlib scikit-learn scipy
jupyter notebook hierarchical_dbscan_analysis.ipynb
```

The dataset ships with scikit-learn, so nothing needs downloading. Run all cells top to bottom. The PCA step uses `random_state=42` so results reproduce exactly.
