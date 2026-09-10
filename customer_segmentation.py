import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("Mall_Customers.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())


# ==========================================
# 2. REMOVE CUSTOMER ID
# ==========================================

df = df.drop("CustomerID", axis=1)

print("\nAfter Removing CustomerID:")
print(df.head())


# ==========================================
# 3. DATA VISUALIZATION
# ==========================================

# Age Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Age"], bins=10, kde=True)
plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.savefig("age_distribution.png")
plt.close()


# Annual Income Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Annual Income (k$)"], bins=10, kde=True)
plt.title("Annual Income Distribution")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Number of Customers")
plt.savefig("income_distribution.png")
plt.close()


# Spending Score Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Spending Score (1-100)"], bins=10, kde=True)
plt.title("Spending Score Distribution")
plt.xlabel("Spending Score")
plt.ylabel("Number of Customers")
plt.savefig("spending_distribution.png")
plt.close()


# ==========================================
# 4. SELECT FEATURES
# ==========================================

X = df[[
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]]

print("\nSelected Features:")
print(X.head())


# ==========================================
# 5. SCALE DATA
# ==========================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nScaled Data:")
print(X_scaled[:5])


# ==========================================
# 6. ELBOW METHOD
# ==========================================

wcss = []

for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)


plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.xticks(range(1, 11))
plt.savefig("elbow_method.png")
plt.close()


# ==========================================
# 7. APPLY K-MEANS
# ==========================================

# Elbow Method shows K = 4
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)


# ==========================================
# 8. DISPLAY CLUSTERS
# ==========================================

print("\nCustomer Clusters:")
print(df.head(10))


print("\nNumber of Customers in Each Cluster:")
print(df["Cluster"].value_counts().sort_index())

# ==========================================
# 9. CLUSTER VISUALIZATION
# ==========================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"],
    s=80
)

# Plot cluster centers
centers = scaler.inverse_transform(kmeans.cluster_centers_)

plt.scatter(
    centers[:, 1],
    centers[:, 2],
    marker="X",
    s=200,
    label="Cluster Centers"
)

plt.title("Customer Segmentation")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.grid(True)

plt.savefig("customer_clusters.png")
plt.show()

# ==========================================
# 10. CLUSTER PROFILING
# ==========================================

cluster_profile = df.groupby("Cluster")[
    ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
].mean()

print("\nCluster Profile:")
print(cluster_profile.round(2))

# ==========================================
# 11. CUSTOMER SEGMENT NAMES
# ==========================================

segment_names = {
    0: "Older Medium-Spending Customers",
    1: "High Income High Spending Customers",
    2: "Young High-Spending Customers",
    3: "High Income Low-Spending Customers"
}

df["Customer Segment"] = df["Cluster"].map(segment_names)

print("\nCustomer Segment Summary:")
print(df[[
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)",
    "Cluster",
    "Customer Segment"
]].head(10))

# ==========================================
# 12. SAVE FINAL DATASET
# ==========================================

df.to_csv("customer_segments.csv", index=False)

print("\nFinal dataset saved as customer_segments.csv")