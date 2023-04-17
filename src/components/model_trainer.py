import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting Training and Test Input Data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "RandomForestClassifier": RandomForestClassifier(),
                "AdaBoostClassifier": AdaBoostClassifier()
            }

            params = {
                "RandomForestClassifier": {
                    'criterion': ['entropy', 'log_loss', 'gini'],
                    'max_features': ['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "AdaBoostClassifier": {
                    'learning_rate': [.1,.01,.5,.001],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }

            model_report: dict = evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,
                                                param=params)
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score <0.6:
                raise CustomException("No Best Model Found")
            
            logging.info("Best Model Found")

            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            accuracy = accuracy_score(y_test,predicted)
            round_accuracy = round(accuracy, 2)

            return round_accuracy*100

        except Exception as e:
            raise CustomException(e,sys)


