# Smart Cart Customer Clustering

## Project Overview

This project performs customer segmentation analysis on retail transaction data using machine learning techniques. The analysis includes data preprocessing, feature engineering, PCA for dimensionality reduction, and K-means clustering to identify distinct customer segments.

## Features

- **Data Preprocessing**: Handling missing values, feature engineering, and data transformation
- **Exploratory Analysis**: Correlation analysis and distribution visualization
- **Dimensionality Reduction**: PCA to reduce features while preserving variance
- **Customer Segmentation**: K-means clustering to identify 4 customer segments:
  - Married Budgeters
  - Premium High-Spenders
  - Young Singles/Browsers
  - Established Mature Families
- **Interactive Visualization**: 3D scatter plots and pairwise comparisons

## Streamlit App

A Streamlit web application is available to interactively explore the customer segments.

### Live Demo

🚀 **[View Live Demo](https://smart-cart-customer-clustering.streamlit.app/)**

Experience the interactive customer segmentation analysis with 3D visualizations, cluster distributions, and PCA insights.

### Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the app:
   ```bash
   streamlit run app.py
   ```

3. Open your browser to `http://localhost:8501`

### Deployment on Streamlit Cloud

1. Create a GitHub repository and upload the following files:
   - `app.py`
   - `requirements.txt`
   - `smartcart_customers (1).csv`
   - `README.md` (optional)

2. Go to [share.streamlit.io](https://share.streamlit.io)

3. Connect your GitHub account and select the repository

4. Deploy the app

## Dataset

The dataset (`smartcart_customers (1).csv`) contains customer demographic and transaction data including:
- Demographic information (age, education, marital status, income)
- Purchase history across different product categories
- Engagement metrics (web visits, recency, response to campaigns)

## Methodology

1. **Data Cleaning**: Impute missing values using KNN imputation
2. **Feature Engineering**: Create total spending, total purchases, age, and tenure features
3. **Transformation**: Apply log transformations to reduce skewness
4. **Encoding**: Label encode education and one-hot encode marital status
5. **Scaling**: Standardize features for PCA
6. **PCA**: Reduce dimensionality while retaining 90% of variance
7. **Clustering**: Apply K-means with 4 clusters
8. **Visualization**: 3D scatter plots and pair plots for segment exploration

## Dependencies

- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- plotly
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
