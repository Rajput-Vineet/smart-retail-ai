import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# =========================
# LOAD DATA
# =========================

rfm = pd.read_csv(
    "data/processed/rfm_scaled.csv"
)

# Features for clustering
X = rfm[["Recency", "Frequency", "Monetary"]]


# =========================
# TRAIN FINAL KMEANS MODEL
# =========================

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

# Create cluster predictions
rfm["Cluster"] = kmeans.fit_predict(X)


# =========================
# MAP BUSINESS LABELS
# =========================

cluster_labels = {
    0: "At Risk Customers",
    1: "Champions",
    2: "Loyal Customers",
    3: "Potential Loyalists"
}

rfm["Segment"] = rfm["Cluster"].map(cluster_labels)


# =========================
# CLUSTER SUMMARY
# =========================

print("=" * 50)
print("CLUSTER SUMMARY")
print("=" * 50)

cluster_summary = rfm.groupby("Segment")[
    ["Recency", "Frequency", "Monetary"]
].mean()

print(cluster_summary)


# =========================
# SEGMENT COUNTS
# =========================

print("\n" + "=" * 50)
print("SEGMENT COUNTS")
print("=" * 50)

print(rfm["Segment"].value_counts())


# =========================
# PCA FOR VISUALIZATION
# =========================

pca = PCA(n_components=2)

pca_features = pca.fit_transform(X)

rfm["PCA1"] = pca_features[:, 0]
rfm["PCA2"] = pca_features[:, 1]


# =========================
# VISUALIZE CLUSTERS
# =========================

plt.figure(figsize=(12, 7))

scatter = plt.scatter(
    rfm["PCA1"],
    rfm["PCA2"],
    c=rfm["Cluster"],
    cmap="viridis",
    alpha=0.7
)

plt.title(
    "Customer Segments (PCA Visualization)",
    fontsize=16
)

plt.xlabel(
    "PCA Component 1",
    fontsize=12
)

plt.ylabel(
    "PCA Component 2",
    fontsize=12
)

plt.colorbar(scatter)

plt.grid(True)

plt.show()


# =========================
# SAVE KMEANS MODEL
# =========================

joblib.dump(
    kmeans,
    "models/segmentation/kmeans_model.pkl"
)

print("\nKMeans model saved successfully!")


# =========================
# SAVE FINAL SEGMENTED DATA
# =========================

rfm.to_csv(
    "data/processed/customer_segments.csv",
    index=False
)

print("\nCustomer segmentation completed!")
print(
    "Segmented customer data saved successfully!"
)