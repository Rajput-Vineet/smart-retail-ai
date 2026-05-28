import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# =========================
# LOAD SCALED DATA
# =========================

rfm = pd.read_csv(
    "data/processed/rfm_scaled.csv"
)

X = rfm[["Recency", "Frequency", "Monetary"]]


# =========================
# ELBOW METHOD
# =========================

inertia = []

K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X)

    inertia.append(kmeans.inertia_)


# =========================
# PLOT ELBOW CURVE
# =========================

plt.figure(figsize=(8, 5))

plt.plot(K_range, inertia, marker='o')

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")

plt.grid(True)

plt.show()


# =========================
# SILHOUETTE SCORES
# =========================

print("\n" + "=" * 50)
print("SILHOUETTE SCORES")
print("=" * 50)

for k in K_range:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X)

    score = silhouette_score(X, labels)

    print(f"K={k} --> Silhouette Score: {score:.4f}")