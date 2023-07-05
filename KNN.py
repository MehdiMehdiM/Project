import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris_data = pd.read_csv(url, names=column_names)

# Split the data into features (X) and label (y)
X = iris_data.drop('species', axis=1)
y = iris_data['species']

# Perform feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Create end point using FastAPI

from fastapi import FastAPI
from pydantic import BaseModel

# Define the input data model
class PredictionRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Create a FastAPI app instance

app = FastAPI()

# Create the endpoint
@app.post('/knn/predict')
def knn_endpoint(data: IrisData):

    # Input data
    input_data = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    input_data_scaled = scaler.transform(input_data)

    # Make predictions using the trained logistic regression mode
    prediction = knn_model.predict(input_data_scaled)

    # Return
    return {'prediction': prediction[0]}



