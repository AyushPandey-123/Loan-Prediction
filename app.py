from flask import Flask, request, render_template
import numpy as np
import pandas as pd


from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method == "GET":
        return render_template('home.html')
    else:
        data = CustomData(
                Gender = request.form.get("Gender"),
                Married = request.form.get("Married"),
                Dependents = request.form.get("Dependents"),
                Education = request.form.get("Education"),
                Self_Employed = request.form.get("Self_Employed"),
                Applicant_Income = request.form.get("Applicant_Income"),
                Coapplicant_Income = request.form.get("Coapplicant_Income"),
                Loan_Amount = request.form.get("Loan_Amount"),
                Term = request.form.get("Term"),
                Credit_History = request.form.get("Credit_History"),
                Area = request.form.get("Area")
        )

        pred_df = data.get_data_as_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predcit(pred_df)
        if results[0]==0:
            final_result = 'Not Approved'
        else:
            final_result = 'Approved'

        return render_template('home.html',results=final_result)
    
if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True,port=8080)
