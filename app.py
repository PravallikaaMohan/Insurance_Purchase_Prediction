from flask import Flask,request,render_template,jsonify
from pycaret.classification import *
import pandas as pd
import numpy as np

#Initialise the Flask app
app = Flask(__name__)

#Load pre-trained model
model = load_model('MODEL')
cols = ['age', 'sex', 'bmi', 'steps', 'children', 'smoker', 'region', 'charges']

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [ x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns=cols)
    prediction = predict_model(model, data=data_unseen, round=0)
    prediction = int(prediction.Label[0])
    if prediction==1:
        output = "will be purchased."
    else:
        output = "will not be purchased."
    return render_template('home.html', pred = "Insurance {}".format(output))


@app.route('/predict_api', methods = ['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
