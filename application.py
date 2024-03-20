from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.pipeline.predict import PredictPipeline

def load_pipeline():
    return PredictPipeline()


application = Flask(__name__)
app=application


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    predict_pipeline=load_pipeline()
    if request.method == 'GET':
        return render_template('home.html')
    else:
        no_dependents = int(request.form['no_dependents'])
        income_annum = int(request.form['income_annum'])
        loan_amount = int(request.form['loan_amount'])
        loan_term = int(request.form['loan_term'])
        cibil_score = int(request.form['cibil_score'])
        residential_asset = int(request.form['residential_asset'])
        commercial_asset = int(request.form['commercial_asset'])
        luxury_asset = int(request.form['luxury_asset'])
        bank_asset = int(request.form['bank_asset'])
        education = request.form['education']
        self_employed = request.form['self_employed']

        data = pd.DataFrame({
            "no_of_dependents": [no_dependents],
            "income_annum": [income_annum],
            "loan_amount": [loan_amount],
            "loan_term": [loan_term],
            "cibil_score": [cibil_score],
            "residential_assets_value": [residential_asset],
            "commercial_assets_value": [commercial_asset],
            "luxury_assets_value": [luxury_asset],
            "bank_asset_value": [bank_asset],
            "education": [education],
            "self_employed": [self_employed]
        })

        results = predict_pipeline.make_prediction(data)
        print(results)
        if results[0]==1:
            return render_template('home.html', results='Yes')
        else:
            return render_template('home.html',results='No')

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8080)
