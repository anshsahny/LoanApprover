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


# Home Page
@app.route('/')
def home():
    return render_template('index.html')

# Result Page
@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

# Error Page
@app.route('/error')
def error():
    return render_template('error.html')


if __name__ == '__main__':
    app.run(debug=True)
