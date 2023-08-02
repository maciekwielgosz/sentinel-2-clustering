from sklearn.cluster import KMeans
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# read the dataframe from the csv file
df = pd.read_csv('hr_svd.csv')

# get the SVD component columns
svd_columns = [f'svd_{i}' for i in range(16)]
svd_data = df[svd_columns]

# create a KMeans object and fit it to the SVD data
kmeans = KMeans(n_clusters=6, random_state=0).fit(svd_data)

# get the cluster IDs assigned by the KMeans object
cluster_ids = kmeans.labels_

# add the cluster IDs to the dataframe
df['cluster_id'] = cluster_ids

# save the dataframe to a new csv file
df.to_csv('output_with_clusters.csv', index=False)

# check the dataframe
print(df.head())

# save the dataframe 
df.to_csv('hr_clusterd.csv', index=False)

# Make the plot
plt.figure(figsize=(10, 10))

# Select two dimensions for the x and y axes
sns.scatterplot(x='svd_0', y='svd_1', hue='cluster_id', palette='viridis', data=df)

# Save the plot as a PNG file
plt.savefig('clusters.png')


# Make the plot
plt.figure(figsize=(10, 10))

# Select two dimensions for the x and y axes
sns.scatterplot(x='svd_0', y='svd_1', hue='label_id', palette='viridis', data=df)

# Save the plot as a PNG file
plt.savefig('gt_clusters.png')

# The purity of a cluster is the maximum number of data points from a single class 
# in a cluster divided by the total number of data points in that cluster. 
# To compute the purity of each cluster in your DataFrame, you can group the DataFrame 
# by 'cluster_id' and 'label_id', count the size of each group, and then divide by the size of the cluster.


def compute_purity(df):
    # Get the number of data points from the largest class in each cluster
    cluster_label_counts = df.groupby(['cluster_id', 'label_id']).size()
    max_counts = cluster_label_counts.groupby('cluster_id').max()
    
    # Get the total number of data points in each cluster
    cluster_counts = df['cluster_id'].value_counts()
    
    # Divide the two to get purity
    purity = max_counts / cluster_counts
    
    return purity

purity = compute_purity(df)

print("Cluster id, Purity: ",  purity)


