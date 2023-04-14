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
print(df.columns)
