from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import logging
import joblib
import json
import sys
import os

current_dir = os.path.dirname(__file__)

app = Flask(__name__, static_folder='static', template_folder='template')

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)


# Function that handles the ML prediction
def prediction(data=pd.DataFrame):
    # Get model and file path
    model_name = 'bin/xgboostModel.pkl'
    model_dir = os.path.join(current_dir, model_name)
    # Load the model
    loaded_model = joblib.load(open(model_dir, 'rb'))
    # Predict the result
    result = loaded_model.predict(data)
    # Return the result
    return result[0]


# Home Page
@app.route('/')
def home():
    return render_template('index.html')


# Prediction Page
@app.route('/prediction', methods=['POST'])
def predict():
    if request.method == 'POST':
        return render_template('prediction.html')
    else:
        # Return Error Page
        return render_template('error.html')


if __name__ == '__main__':
    app.run(debug=True)
