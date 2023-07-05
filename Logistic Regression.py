import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris_data = pd.read_csv(url, names=column_names)

# Split the data into features (X) and label (y)
X = iris_data.drop('species', axis=1)
y = iris_data['species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
lg_model = LogisticRegression()
lg_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lg_model.predict(X_test)

# Accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Create end point using FastAPI

from fastapi import FastAPI
from pydantic import BaseModel

# Define the input data model
class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Create a FastAPI app instance

app = FastAPI()

# Define the endpoint
@app.post("/logistic_regression")
def logistic_regression(data: IrisData):

    # Input data
    input_data = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]

    # Make predictions using the trained logistic regression model
    predictions = lg_model.predict(input_data)

    # Return
    return {'prediction': predictions[0]}