from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('admission_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [
        float(request.form['ssc']),
        float(request.form['hsc']),
        float(request.form['degree']),
        float(request.form['entrance'])
    ]
    prediction = model.predict([np.array(features)])
    return render_template('result.html', result=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)

