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
        # Load the medians
        medians = joblib.load('model/medians.pkl')
        median_size = medians['median_size']
        median_grade = medians['median_grade']
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

        # Recreate the features with preprocessing
        aggressive = 1 if (size > median_size and grade > median_grade) else 0
        features = np.array([[age, meno, size, grade, nodes, pgr, er, hormon, rfstime, aggressive]])

        # Predict using the loaded model
        prediction = model.predict(features)[0]

        # Map prediction to readable text
        result = "There is a 90% likelihood of surviving without cancer recurrence." if prediction == 0 else "There is a 90% likelihood of cancer recurrence or mortality."

        return render_template('index.html', prediction=result, patient_name=request.form['name'], date=request.form['date'])
    except KeyError as e:
        return f"Missing or incorrect form field: {str(e)}", 400
    except Exception as e:
        return f"An error occurred: {str(e)}", 500
