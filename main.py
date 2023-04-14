from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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

# choose the number of clusters to create using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++',
                    max_iter=300, n_init=10, random_state=0)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)
print(wcss)
