# Exploring-pandas

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# Manually inputting the dataset based on the provided data
data_dict = {
    "Study_Hours_Per_Day": [6.9, 5.3, 5.1, 6.5, 8.1],
    "Extracurricular_Hours_Per_Day": [3.8, 3.5, 3.9, 2.1, 0.6],
    "Sleep_Hours_Per_Day": [8.7, 8.0, 9.2, 7.2, 6.5],
    "Social_Hours_Per_Day": [2.8, 4.2, 1.2, 1.7, 2.2],
    "Physical_Activity_Hours_Per_Day": [1.8, 3.0, 4.6, 6.5, 6.6],
    "GPA": [2.99, 2.75, 2.67, 2.88, 3.51],
    "Stress_Level": ["Moderate", "Low", "Low", "Moderate", "High"]
}
df = pd.DataFrame(data_dict)

# Splitting data into features (X) and targets (y)
X = df[["Study_Hours_Per_Day", "Extracurricular_Hours_Per_Day", "Sleep_Hours_Per_Day", 
        "Social_Hours_Per_Day", "Physical_Activity_Hours_Per_Day"]]
y_gpa = df["GPA"]
y_stress = df["Stress_Level"]

# Encoding Stress_Level for classification
le = LabelEncoder()
y_stress_encoded = le.fit_transform(y_stress)

# Splitting data for regression and classification tasks
X_train, X_test, y_gpa_train, y_gpa_test = train_test_split(X, y_gpa, test_size=0.4, random_state=42)
_, _, y_stress_train, y_stress_test = train_test_split(X, y_stress_encoded, test_size=0.4, random_state=42)

# 1. Linear Regression (Predict GPA)
lr = LinearRegression()
lr.fit(X_train, y_gpa_train)
y_gpa_pred = lr.predict(X_test)
linear_regression_mse = mean_squared_error(y_gpa_test, y_gpa_pred)

# 2. Logistic Regression (Classify Stress_Level)
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_stress_train)
y_stress_pred_logistic = log_reg.predict(X_test)
logistic_regression_accuracy = accuracy_score(y_stress_test, y_stress_pred_logistic)

# 3. KNN (Classify Stress_Level)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_stress_train)
y_stress_pred_knn = knn.predict(X_test)
knn_accuracy = accuracy_score(y_stress_test, y_stress_pred_knn)

# 4. Clustering (KMeans for grouping)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)
cluster_centers = kmeans.cluster_centers_

# Print the results
print("Linear Regression Mean Squared Error (MSE):", linear_regression_mse)
print("Logistic Regression Accuracy:", logistic_regression_accuracy)
print("KNN Accuracy:", knn_accuracy)
print("KMeans Clusters:", clusters)
print("KMeans Cluster Centers:\n", cluster_centers)
