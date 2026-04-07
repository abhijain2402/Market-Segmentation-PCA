import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Smart Cart Customer Clustering",
    layout="wide",
    page_icon="🛒"
)

# Title
st.title("Smart Cart Customer Segmentation Analysis")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("smartcart_customers (1).csv")
    return df

df = load_data()

st.header("Data Overview")
st.write("Dataset shape:", df.shape)
st.dataframe(df.head())

# Impute missing values
imputer = KNNImputer(n_neighbors=5)
columns_to_impute = ["Income","MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds"]
x = df[columns_to_impute].values
x_imputed = imputer.fit_transform(x)
df["Income"] = x_imputed[:, 0]

# Feature engineering
df["total_spent"] = df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"] + df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
df["total_purchases"] = df[["NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]].sum(axis=1)
df["AGE"] = 2026 - df["Year_Birth"]
df["tenure"] = (pd.to_datetime("01-03-2026") - pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y")).dt.days

# Filter age
df = df[df["AGE"] <= 90]

# Log transformations
spend_cols = ['Income', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'total_purchases']
for col in spend_cols:
    df[col + '_log'] = np.log1p(df[col])

# Encoding
Graduation_mapping = {"Basic": 0, "2n Cycle": 1, "Graduation": 1, "Master": 2, "PhD": 3}
df["education_encoded"] = df["Education"].replace(Graduation_mapping).infer_objects(copy=False)
marital_dummies = pd.get_dummies(df["Marital_Status"], prefix="Marital", drop_first=True, dtype=int)
df = pd.concat([df, marital_dummies], axis=1)

# Select features for clustering
corr_columns = ["total_purchases","MntGoldProds_log","MntSweetProducts_log","MntFishProducts_log","MntMeatProducts_log","MntFruits_log","MntWines_log","Income_log","tenure","AGE","total_spent","Response","Complain","NumWebVisitsMonth","Recency","Teenhome","Kidhome","Marital_YOLO","Marital_Widow","Marital_Together","Marital_Single","Marital_Married","Marital_Alone","education_encoded"]

# Ensure all columns exist (handle missing marital status columns)
existing_corr_columns = [col for col in corr_columns if col in df.columns]
x = df[existing_corr_columns]

# Scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# PCA
pca = PCA()
pca_components = pca.fit_transform(x_scaled)

# Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(pca_components)

# Create dataframe with PCA components and clusters
df_pca = pd.DataFrame(pca_components[:, :3], columns=["PC1", "PC2", "PC3"])
df_pca["cluster"] = clusters

# Cluster names
cluster_names = {
    0: 'Married Budgeters',
    1: 'Premium High-Spenders',
    2: 'Young Singles/Browsers',
    3: 'Established Mature Families'
}
df_pca['cluster_name'] = df_pca['cluster'].map(cluster_names)

st.header("Clustering Results")

# Cluster distribution
st.subheader("Cluster Distribution")
cluster_counts = df_pca["cluster_name"].value_counts()
st.bar_chart(cluster_counts)

# 3D Visualization
st.subheader("3D Customer Segments Visualization")
fig = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3',
                    color='cluster_name',
                    title='Customer Segments in 3D Space',
                    opacity=0.8)
st.plotly_chart(fig)

# Pairplot
st.subheader("Pairwise Cluster Comparison")
cols_to_plot = ['PC1', 'PC2', 'PC3', 'cluster_name']
g = sns.pairplot(df_pca[cols_to_plot], hue='cluster_name', palette='magma', diag_kind='kde', plot_kws={'alpha': 0.6, 's': 30})
plt.suptitle('Pairwise Comparison of PCA Clusters', y=1.02)
st.pyplot(g.fig)

# PCA explained variance
st.header("PCA Analysis")
explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)

st.subheader("Explained Variance")
for i, (e, c) in enumerate(zip(explained, cumulative)):
    st.write(f"PC{i+1}: {e:.1%} explained | {c:.1%} cumulative")

# Variance plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.bar(range(1, len(explained)+1), explained*100, color="purple", edgecolor="black")
ax1.set_xlabel("Principal Components")
ax1.set_ylabel("Explained Variance (%)")
ax1.set_title("PCA Individual Explained Variance")

ax2.plot(range(1, len(cumulative)+1), cumulative*100, color="blue", marker="o")
ax2.set_xlabel("Principal Components")
ax2.set_ylabel("Cumulative Explained Variance (%)")
ax2.set_title("PCA Cumulative Explained Variance")
ax2.axhline(y=80, color="red", linestyle="--", label="80% threshold")
ax2.axhline(y=90, color="green", linestyle="--", label="90% threshold")
ax2.legend()
plt.tight_layout()
st.pyplot(fig)

st.success("Analysis complete! This Streamlit app showcases customer segmentation using K-means clustering on PCA-transformed features.")

st.success("Analysis complete! This Streamlit app showcases customer segmentation using K-means clustering on PCA-transformed features.")

# Footer
st.markdown("---")
st.markdown("### 👨‍💻 About the Developer")

# Developer info in columns
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    **Abhijain**  
    *Data Science & Machine Learning Enthusiast*

    Passionate about transforming data into actionable insights through machine learning and interactive visualizations.
    This project demonstrates expertise in customer analytics, clustering algorithms, and web app development.

    **Skills:** Python, Machine Learning, Data Visualization, Streamlit, Scikit-learn, Pandas
    """)

with col2:
    st.markdown("**Connect with me:**")
    st.markdown("""
    <div style="display: flex; flex-direction: column; gap: 10px;">
        <a href="https://www.linkedin.com/in/abhi-jain-901a42285" target="_blank" style="text-decoration: none; color: #0077B5; font-weight: bold;">
            🔗 LinkedIn
        </a>
        <a href="https://github.com/abhijain2402" target="_blank" style="text-decoration: none; color: #333; font-weight: bold;">
            💻 GitHub
        </a>
        <a href="mailto:abhijain905@gmail.com" style="text-decoration: none; color: #EA4335; font-weight: bold;">
            📧 Email
        </a>
    </div>
    """, unsafe_allow_html=True)