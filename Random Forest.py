import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris_data = pd.read_csv(url, names=column_names)

# Split the dataset into features (X) and label (y)
X = iris_data.drop('species', axis=1)
y = iris_data['species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

# Evaluate and make predictions on the test set
y_pred = rf_classifier.predict(X_test)
print(classification_report(y_test, y_pred))

# Create endpoint using FastAPI

from fastapi import FastAPI
from pydantic import BaseModel

# Create a FastAPI app instance

app = FastAPI()

# Define the input data model
class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Create the endpoint
@app.post("/random-forest")
def random_forest_endpoint(data: IrisData):

    # Input data
    input_data = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]

    # Use the trained Random Forest Classifier to make predictions
    prediction = rf_classifier.predict(input_data)

    # Return the predicted class
    return {"prediction": prediction[0]}






