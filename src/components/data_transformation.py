import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join('artifacts','preprocesor.pkl')

class DataTransformation:

    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer(self):
        try:
            numerical_columns = ['Total_Income', 'Loan_Amount', 'Term', 'Credit_History', 'Status']

            categorical_column = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Area']

            num_pipeline = Pipeline(
                steps=[
                    [("Imputer",SimpleImputer(strategy='median')),
                     ("Scaler",StandardScaler())
                     ]
                ]
            )
            cat_pipeline = Pipeline(
                steps = [
                    ("Imputer",SimpleImputer(strategy="most_frequent")),
                    ("One_Hot_Encoder",OneHotEncoder()),
                ]
            )

            logging.info("Numerical Columns Encoding Completed")

            logging.info("Categorical Columns Encoding Completed")

            preprocessor = ColumnTransformer(
                [
                    ("Num_Pipeline",num_pipeline,numerical_columns),
                    ("Categorical_Pipeline",cat_pipeline,categorical_column)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
            

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and Test Data Readed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer()

            target_column = 'Status'

            input_features_train_df = train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df = train_df[target_column]

            input_features_test_df = test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info("Applying Preprocessing object on training and testing dataframe")

            train_arr = np.c_[
                input_features_train_df,np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_features_test_df,np.array(target_feature_test_df)
            ]

            logging.info("Saved Preprocessing Object")

            save_obj(
                file_path = self.data_transformation_config.preprocessor_ob_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)