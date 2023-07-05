import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI
from pydantic import BaseModel

# Load the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris_data = pd.read_csv(url, names=column_names)

# Split the dataset into features (X) and label (y)
X = iris_data.drop('species', axis=1)
y = iris_data['species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Model
svm_model = SVC()
svm_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test_scaled)

# Accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Create endpoint using FastAPI

# Create a FastAPI app instance

app = FastAPI()

# Define a request body model
class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Create the endpoint
@app.post("/svm-predict")
def svm_endpoint(data: IrisData):

    # Input data
    input_data = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    input_data_scaled = scaler.transform(input_data)

    # Make predictions using the trained SVM model
    predictions = svm_model.predict(input_data_scaled)

    # Return
    return {"prediction": predictions[0]}


