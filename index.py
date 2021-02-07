# A classification ML Model. Utilizes K-Neighbors, with a loop to determine the optimal knn.
from flask import Flask
import pandas as pd
import numpy as np

# import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.impute import KNNImputer
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)


# Import dataset, sample shape: (10459, 35)
df = pd.read_csv('RiskData.csv')


# Clean dataset (filled with NaN values to impute).
df.replace('.', np.nan, inplace=True)


# Create arrays for the features and the response variable
y = df['Risk_Flag'].values
X = df.drop('Risk_Flag', axis=1).values


"""
Setup the pipeline: 
    1. imputes knn values, 
    2. scales data, 
    3. nusvc as opposed to svc or linearsvc
"""
steps = [('imputation', KNNImputer(missing_values=np.nan, n_neighbors=6)),
         ('standardscaler', StandardScaler()),
         ('nusvc', NuSVC())]

pipeline = Pipeline(steps)


# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)


# Fit the pipeline to the train set
pipeline.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = pipeline.predict(X_test)


"""
Compute metrics: 
    1. confusion matrix, 
    2. classification report, 
    3. model score.
"""
print(f"\n\n Confusion Matrix: \n {str(confusion_matrix(y_test, y_pred))} \n")
print(f"Classification Report: \n {str(classification_report(y_test, y_pred))} \n")
print(f"Model Score: {str(pipeline.score(X_test, y_test))} \n")


# Basic route, why not. Not ideal.
@app.route('/')
def hello_classification():
    return f"Model Score: {str(pipeline.score(X_test, y_test))}"