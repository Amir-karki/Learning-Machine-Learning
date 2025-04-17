import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# import custom made KNN class from knn_from_scratch
from knn_from_scratch import KNN

# load the dataset
df = pd.read_csv('Social_Network_Ads.csv')

# drop userId as it is of no use
df.drop(columns=['User ID'], inplace=True)
# print(df.head(5))

# change the categorical column to Numerical with LabelEncoder
encoder = LabelEncoder()
df["Gender"] = encoder.fit_transform(df["Gender"])
print(df.head(5))

# Extract features(X) and labels(y)
X = df.iloc[:, 1:3].values
y = df.iloc[:, -1].values

# Scale the numerical column to bring it to a same scale
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
# print(X)

# split the dataset into trainig and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape)
print(X_test[0:3])

# Train and test the model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
pred = knn_model.predict(X_test)
print(accuracy_score(y_test, pred))

# Initialize custom made KNN
custom_knn = KNN(k=5)
custom_knn.fit(X_train, y_train)
y_pred = custom_knn.predict(X_test)
print("Accuracy from custom KNN class")
print(accuracy_score(y_test, y_pred))





