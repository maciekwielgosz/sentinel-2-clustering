from sklearn.cluster import KMeans
import pandas as pd


# read the dataframe from the csv file
df = pd.read_csv('hr_svd.csv')

# get the SVD component columns
svd_columns = [f'svd_{i}' for i in range(16)]
svd_data = df[svd_columns]

# create a KMeans object and fit it to the SVD data
kmeans = KMeans(n_clusters=5, random_state=0).fit(svd_data)

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

