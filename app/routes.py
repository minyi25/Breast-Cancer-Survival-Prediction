from flask import request, jsonify, render_template
import joblib
import numpy as np

from . import create_app

app = create_app()

# Load pre-trained model
model = joblib.load('model/breast_cancer_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input data
        data = request.json
        features = np.array([[
            data['age'], data['meno'], data['size'], data['grade'],
            data['nodes'], data['pgr'], data['er'], data['hormon'], data['rfstime']
        ]])
        prediction = model.predict(features)[0]
        return jsonify({'status': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

