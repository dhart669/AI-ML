import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Generate synthetic customer data
np.random.seed(42)
num_customers = 200

# Features: Age, Annual Income, Spending Score
age = np.random.randint(18, 70, size=num_customers)
income = np.random.randint(15000, 100000, size=num_customers)
spending_score = np.random.randint(1, 100, size=num_customers)

# Create a DataFrame
data = pd.DataFrame({
    'Age': age,
    'Annual Income': income,
    'Spending Score': spending_score
})

# Display the first few rows of the dataset
print(data.head())

# Normalize the data
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# Apply K-Means clustering
k = 5  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(data_normalized)
labels = kmeans.labels_

# Add cluster labels to the original data
data['Cluster'] = labels

# Evaluate clustering performance using silhouette score
sil_score = silhouette_score(data_normalized, labels)
print(f"Silhouette Score: {sil_score:.2f}")

# Plotting the clusters
fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(121)
scatter = ax1.scatter(data['Annual Income'], data['Spending Score'], c=data['Cluster'], cmap='viridis')
ax1.set_title('Customer Segments')
ax1.set_xlabel('Annual Income')
ax1.set_ylabel('Spending Score')

# Create a legend for clusters
legend1 = ax1.legend(*scatter.legend_elements(), title="Clusters")
ax1.add_artist(legend1)

ax2 = fig.add_subplot(122)
scatter = ax2.scatter(data['Age'], data['Spending Score'], c=data['Cluster'], cmap='viridis')
ax2.set_title('Customer Segments')
ax2.set_xlabel('Age')
ax2.set_ylabel('Spending Score')

# Create a legend for clusters
legend2 = ax2.legend(*scatter.legend_elements(), title="Clusters")
ax2.add_artist(legend2)

plt.show()

