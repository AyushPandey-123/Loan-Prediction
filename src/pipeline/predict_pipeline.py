import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predcit(self,features):
        
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocesor.pkl'
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path= preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            print(preds)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData():
    def __init__(self,
                 Gender:str,
                 Married:str,
                 Dependents:int,
                 Education:str,
                 Self_Employed:str,
                 Applicant_Income:int,
                 Coapplicant_Income:int,
                 Loan_Amount:int,
                 Term:int,
                 Credit_History:int,
                 Area:str,
                 ):

        self.Gender = Gender

        self.Married = Married

        self.Dependents = Dependents

        self.Education = Education

        self.Self_Employed = Self_Employed

        self.Applicant_Income = Applicant_Income

        self.Coapplicant_Income = Coapplicant_Income

        self.Loan_Amount = Loan_Amount

        self.Term = Term
        
        self.Credit_History = Credit_History

        self.Area = Area

    def get_data_as_frame(self):
        try:
            custom_data_input_dict = {
                "Gender": [self.Gender],
                "Married": [self.Married],
                "Dependents": [self.Dependents],
                "Education": [self.Education],
                "Self_Employed": [self.Self_Employed],
                "Applicant_Income": [self.Applicant_Income],
                "Coapplicant_Income": [self.Coapplicant_Income],
                "Loan_Amount": [self.Loan_Amount],
                "Term": [self.Term],
                "Credit_History": [self.Credit_History],
                "Area": [self.Area]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)

