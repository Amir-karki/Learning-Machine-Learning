import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from Kmeans import KMeans
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

# centroids = [(-5, -5), (5, 5), (-2, -2)]
# cluster_std = [1, 1, 1]

# X, y = make_blobs(n_samples=100, cluster_std=cluster_std, centers=centroids,
#                   n_features=2, random_state=2)

# testing our code with imported data
df = pd.read_csv('C:/Users/12368/MyMLRepo/KMeans_From_Scratch/student_clustering.csv')

X = df.iloc[:, :].values

km = KMeans(n_clusters=3, max_iter=250)
y_means = km.fit_predict(X)

# print(y_means)

plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1])
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1])
plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1])


plt.show()