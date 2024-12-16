import joblib
import numpy as np
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Load the pre-trained model
model_path = 'model/breast_cancer_model.pkl'
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')
    
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Debugging: Print incoming form data
        print("Form Data Received:", request.form)

        # Extract input data
        age = float(request.form['age'])
        meno = int(request.form['meno'])
        size = float(request.form['size'])
        grade = int(request.form['grade'])
        nodes = int(request.form['nodes'])
        pgr = float(request.form['pgr'])
        er = float(request.form['er'])
        hormon = int(request.form.get('hormon', 0))
        rfstime = float(request.form['rfstime'])

        # Create feature array
        features = np.array([[age, meno, size, grade, nodes, pgr, er, hormon, rfstime]])

        # Make prediction
        prediction = model.predict(features)[0]

        # Map prediction to human-readable result
        result = "Alive without recurrence" if prediction == 0 else "Recurrence or death"
        return render_template('index.html', prediction=result)

    except Exception as e:
        print("Error:", e)
        return str(e), 400
