# app.py
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("diabetes_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        features = [float(x) for x in request.form.values()]
        input_data = np.array([features])

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Convert to readable text
        result = "Positive for Diabetes 😔" if prediction == 1 else "Negative for Diabetes 😊"
        return render_template('index.html', prediction_text=f'Result: {result}')
    except:
        return render_template('index.html', prediction_text="⚠️ Please fill all fields correctly.")

if __name__ == "__main__":
    app.run(debug=True)
