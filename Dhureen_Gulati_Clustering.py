import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.decomposition import PCA

# File paths
transac_path = '/Users/dhureengulati/Downloads/Transactions.csv'
prod_path = '/Users/dhureengulati/Downloads/Products.csv'
cust_path = '/Users/dhureengulati/Downloads/Customers.csv'

# Load data
transac_df = pd.read_csv(transac_path)
prod_df = pd.read_csv(prod_path)
cust_df = pd.read_csv(cust_path)

# Convert dates to datetime format
cust_df['SignupDate'] = pd.to_datetime(cust_df['SignupDate'])
transac_df['TransactionDate'] = pd.to_datetime(transac_df['TransactionDate'])

# Aggregate transaction data by CustomerID
customer_transactions = transac_df.groupby('CustomerID').agg(
    TotalSpending=('TotalValue', 'sum'),
    PurchaseFrequency=('TransactionID', 'count'),
    ProductDiversity=('ProductID', pd.Series.nunique)
).reset_index()

# Merge aggregated data with customer profile data
customer_profile = pd.merge(cust_df, customer_transactions, on='CustomerID', how='left')
customer_profile.fillna({'TotalSpending': 0, 'PurchaseFrequency': 0, 'ProductDiversity': 0}, inplace=True)

# Selecting numerical features for clustering
features = customer_profile[['TotalSpending', 'PurchaseFrequency', 'ProductDiversity']]

# Normalizing the data
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)

# Perform clustering
num_clusters = 4  # Changeable based on experimentation
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(normalized_features)
customer_profile['Cluster'] = kmeans.labels_

# Calculate clustering metrics
db_index = davies_bouldin_score(normalized_features, kmeans.labels_)
silhouette_avg = silhouette_score(normalized_features, kmeans.labels_)

# Visualize clusters using PCA
pca = PCA(n_components=2)
pca_features = pca.fit_transform(normalized_features)
customer_profile['PCA1'] = pca_features[:, 0]
customer_profile['PCA2'] = pca_features[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='PCA1', y='PCA2', hue='Cluster', palette='viridis', data=customer_profile, s=50
)
plt.title('Customer Clusters (PCA Reduced)', fontsize=16)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.show()

# Cluster profiles
cluster_summary = customer_profile.groupby('Cluster')[['TotalSpending', 'PurchaseFrequency', 'ProductDiversity']].mean()
print("Cluster Summary:")
print(cluster_summary)

# Print clustering metrics
print(f"Number of Clusters: {num_clusters}")
print(f"Davies-Bouldin Index: {db_index:.3f}")
print(f"Silhouette Score: {silhouette_avg:.3f}")
