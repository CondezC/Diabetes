import sys
import joblib
import json

# Load the trained model
model = joblib.load("diabetes_model.pkl")

# Get input values from PHP (JSON string)
data = json.loads(sys.argv[1])

# Predict
prediction = model.predict([data])

# Print result for PHP
print(int(prediction[0]))
