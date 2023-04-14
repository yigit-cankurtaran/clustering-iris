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

# Visualize the results
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 6))
ax1.set_title('K Means')
ax1.scatter(df['sepal length (cm)'], df['sepal width (cm)'],
            c=df['cluster'], cmap='rainbow')
ax2.set_title("Original")
ax2.scatter(df['sepal length (cm)'], df['sepal width (cm)'],
            c=iris.target, cmap='rainbow')
plt.show()

# Predict the cluster of a new data point
new_data = np.array([[5.1, 3.5, 1.4, 0.2]])
new_data_scaled = scaler.transform(new_data)
print(kmeans.predict(new_data_scaled))
