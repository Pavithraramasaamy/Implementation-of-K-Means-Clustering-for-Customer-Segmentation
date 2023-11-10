# EX-08 Implementation of K Means Clustering for Customer Segmentation.

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Choose the number of clusters (K).

2. Decide how many clusters you want to identify in your data.

3. Initialize cluster centroids: Randomly select K data points from your dataset as the initial centroids of the clusters.

4. Assign data points to clusters: Calculate the distance between each data point and each centroid.

5. If the evaluation metrics allow for it, you can compare the results of multiple clustering runs with different K values and select the one that best suits your requirements.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: PAVITHRA R
RegisterNumber: 212222230106
*/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("Mall_Customers (1).csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.cluster import KMeans
wcss=[]

for i in range (1,11):
    kmeans=KMeans(n_clusters = i,init="k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("No. of clusters")
plt.ylabel("wcss")
plt.title("Elbow matter")

km=KMeans(n_clusters=5)
km.fit(data.iloc[:,3:])

y_pred=km.predict(data.iloc[:,3:])
y_pred

data["cluster"]=y_pred
df0=data[data["cluster"]==0]
df1=data[data["cluster"]==1]
df2=data[data["cluster"]==2]
df3=data[data["cluster"]==3]
df4=data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="black",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="blue",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="magenta",label="cluster4")
plt.legend()
plt.title("Customer Segmets")
```

## Output:
## DATASET:

![1](https://github.com/Pavithraramasaamy/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118596964/8c29a6e1-9f13-49ec-ad7f-52a2a6ceae0b)


## data.head().
![2](https://github.com/Pavithraramasaamy/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118596964/28f86f70-eaaa-49a5-8534-b8c8e69d7d31)


## data.info().

![3](https://github.com/Pavithraramasaamy/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118596964/91a49a82-871f-4aa3-8c9f-75644fd911d2)

## data.isnull().sum().

![4](https://github.com/Pavithraramasaamy/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118596964/723f23ce-744b-4357-8873-6c48a81e17a6)

## Elbow method graph.

![5](https://github.com/Pavithraramasaamy/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118596964/af124f8a-c6a7-42a7-8623-4e3bcd2a1ab7)

## K Means clusters.

![6](https://github.com/Pavithraramasaamy/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118596964/c1e5241d-e46f-4a0b-ba7d-a2034dcdb4e6)

## Y_prediction value.

![7](https://github.com/Pavithraramasaamy/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118596964/7209528d-b484-4b0c-9961-9f3af99f1eb9)

## Customers Segments Graph.

![8](https://github.com/Pavithraramasaamy/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118596964/3aa6d9ca-b7a2-4d93-8ac2-98a4e2ccaaea)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
