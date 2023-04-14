from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Load the iris dataset
iris = load_iris()
X = iris.data

# Load the data into a pandas dataframe
df = pd.DataFrame(X, columns=iris.feature_names)

# Feature engineering to improve the model
df['petal ratio'] = df['petal length (cm)'] / df['petal width (cm)']
df['sepal ratio'] = df['sepal length (cm)'] / df['sepal width (cm)']

# Separate the columns you want to scale from the new features
cols_to_scale = ['sepal length (cm)', 'sepal width (cm)',
                 'petal length (cm)', 'petal width (cm)']
scaled_df = df[cols_to_scale]

# Scale the dataframe
scaler = StandardScaler()
scaled_data = scaler.fit_transform(scaled_df)

# Add the new features back in
scaled_df = pd.DataFrame(scaled_data, columns=cols_to_scale)
scaled_df['petal ratio'] = df['petal ratio']
scaled_df['sepal ratio'] = df['sepal ratio']


# choose the number of clusters using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++',
                    max_iter=300, n_init=10, random_state=0)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

# Plot the elbow method
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
# plt.show()

# the elbow is at 2, so we choose 2 clusters

#  Train the model
kmeans = KMeans(n_clusters=2)
kmeans.fit(scaled_data)

df['cluster'] = kmeans.labels_

# Visualize the results
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 6))
ax1.set_title('K Means')
ax1.scatter(df['sepal length (cm)'], df['sepal width (cm)'],
            c=df['cluster'], cmap='rainbow')
ax2.set_title("Original")
ax2.scatter(df['sepal length (cm)'], df['sepal width (cm)'],
            c=iris.target, cmap='rainbow')
# plt.show()

#  Predict the cluster of a new data point
new_data = np.array([[5.1, 3.5, 1.4, 0.2]])
new_data_scaled = scaler.transform(new_data)
print(f'New data belongs in: {kmeans.predict(new_data_scaled)}')
# returns [0]

# in unsupervised learning, we don't have labels, so we can't evaluate the model
# we can use the silhouette score to evaluate the model

labels = kmeans.labels_
silhouette_avg = silhouette_score(scaled_data, labels, metric='cosine')
print(f'Silhouette score is: {silhouette_avg}')
# using the euclidean distance, the silhouette score is 0.58, not very good
# manhattan distance returned 0.62
# because the data is normalized, cosine distance returned 0.73, which is the best
