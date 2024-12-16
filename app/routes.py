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
        # Extract input data
        age = float(request.form['age'])
        meno = int(request.form['meno'])
        size = float(request.form['size'])
        grade = int(request.form['grade'])
        nodes = int(request.form['nodes'])
        pgr = float(request.form['pgr'])
        er = float(request.form['er'])
        hormon = int(request.form['hormon'])
        rfstime = float(request.form['rfstime'])

        # Recreate the features with preprocessing
        features = np.array([[age, meno, size, grade, nodes, pgr, er, hormon, rfstime]])

        # Create the "aggressive" feature (if used in training)
        aggressive = np.where((size > X_train['size'].median()) & (grade > X_train['grade'].median()), 1, 0)
        features = np.append(features, aggressive).reshape(1, -1)

        # Predict using the loaded model
        prediction = model.predict(features)[0]

        # Map prediction to readable text
        result = "Alive without recurrence" if prediction == 0 else "Recurrence or death"

        return render_template('index.html', prediction=result)
    except KeyError as e:
        return f"Missing or incorrect form field: {str(e)}", 400
    except Exception as e:
        return f"An error occurred: {str(e)}", 500
