from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the iris dataset
iris = load_iris()
X = iris.data

#  Load the data into a pandas dataframe
df = pd.DataFrame(X, columns=iris.feature_names)
# print(df.columns)

#  Scale the dataframe
scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)
# print(scaled_data)

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
plt.show()

# the elbow is at 2, so we choose 2 clusters

#  Train the model
kmeans = KMeans(n_clusters=2)
kmeans.fit(scaled_data)

df['cluster'] = kmeans.labels_

#  Plot the clusters with a histogram
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]

plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'],
            color='green', label='Iris-setosa')
plt.scatter(df2['sepal length (cm)'], df2['sepal width (cm)'],
            color='red', label='Iris-versicolour')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            color='black', label='Centroids')
plt.legend()
plt.show()
