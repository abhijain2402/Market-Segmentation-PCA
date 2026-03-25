# Market-Segmentation-PCA

## Project Overview

This project performs customer analysis and clustering on retail transaction data using a sample dataset (`smartcart_customers (1).csv`).

Goals:
- Clean and preprocess customer data
- Engineer features for customer value and behavior
- Use PCA for dimensionality reduction and exploration
- Apply clustering (K-Means) to identify customer segments
- Visualize results in 2D/3D and interpret each cluster

## Files

- `smartcart_customers (1).csv`: source dataset
- `smartcart_customers_clusters.ipynb`: main notebook with full EDA, feature engineering, PCA, and clustering pipeline
- `README.md`: project guide and roadmap

## Roadmap (Steps + Why)

1. Data Loading and Basic Inspection
   - Load CSV into DataFrame
   - `df.info()`, `df.describe()` and head checks
   - Why: verify completeness, types, missing values, and initial distributions

2. Data Cleaning and Preprocessing
   - Handle missing values (KNN imputer for numeric fields)
   - Filter edge records (e.g., remove unrealistic age > 90)
   - Why: ensures model input quality and reduces bias from invalid values

3. Feature Engineering
   - Create `total_spent`, `total_purchases`, `AGE`, `tenure` from existing fields
   - Derive log transforms for skewed variables (`Income_log`, `MntWines_log`, etc.)
   - Why: capture full spending behavior and correct heavy-tailed distributions

4. Categorical Encoding
   - Map education to numeric with `Graduation_mapping`
   - One-hot encode marital status (with drop_first)
   - Why: prepare categorical fields for PCA/clustering without semantic loss

5. Correlations and Feature Selection
   - Compute correlation matrix on candidate features
   - Choose `corr_columns` for downstream modeling
   - Why: reduce collinearity, choose strong signals, and avoid redundant features

6. Scaling + PCA
   - StandardScaler on selected features
   - Fit PCA to scaled data, compute explained variance
   - Plot explained variances & cumulative ratio to decide n_components
   - Why: normalize, reduce dimension, improve cluster separation

7. Clustering (K-Means)
   - Run K-Means (e.g., `n_clusters=4`) on PCA-transformed data
   - Add cluster labels back to DataFrame
   - Why: find coherent customer groups with similar purchasing behavior

8. Visualization
   - 2D and 3D PCA scatter plots with cluster colors
   - Optional: profile clusters with average metrics in each group
   - Why: interpret segments and validate clustering quality visually

9. Interpretation & Next Steps
   - Summarize segment characteristics (high value, loyal, at-risk, etc.)
   - Suggest business actions (targeted marketing, retention, cross-sell)
   - Why: map technical cluster output to actionable insights

## How to Run

1. Open `smartcart_customers_clusters.ipynb` in Jupyter/VS Code.
2. Execute cells in order from top to bottom.
3. Install dependencies if needed:
   - `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `plotly`
4. Review the cluster summary at the end and export results as CSV if needed.

## Tips

- Plotly 3D may be slow on some machines; use smaller sample for quick preview.
- Use `KMeans(n_clusters=...)` search with silhouette score or elbow method for best cluster count.
- Save the processed dataset separately if you rerun pipeline often.
