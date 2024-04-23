import os 
import sys 
import numpy as np 
import pandas as pd 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from dataclasses import dataclass
from src.Market_Pulse_Analyzing_Popularity_Patterns_for_Apparel_Brands.logger import logging 
from src.Market_Pulse_Analyzing_Popularity_Patterns_for_Apparel_Brands.exception import CustomException
from src.Market_Pulse_Analyzing_Popularity_Patterns_for_Apparel_Brands.utils import save_object
from src.Market_Pulse_Analyzing_Popularity_Patterns_for_Apparel_Brands.utils import model_evaluation
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
import json

@dataclass

class ModelTrainerConfig:
    trained_model_file_path = os.path.join('Artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluation_metric(self,actual,predict):
        accuracyscore = accuracy_score(actual,predict)
        precisionscore = precision_score(actual,predict, average = 'weighted')
        f1score = f1_score(actual,predict, average = 'weighted')
        recallscore = recall_score(actual,predict, average = 'weighted')
        

        return accuracyscore, precisionscore, f1score, recallscore
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Splitting the data into test and train array')
            X_train, y_train, X_test, y_test = (
            train_array[:,:-1],
            train_array[:,-1],
            test_array[:,:-1],
            test_array[:,-1]
              
            )
        
            models = {
            'KNeighborsClassifier' : KNeighborsClassifier(),
            'GaussianNB' : GaussianNB(),
            'SVC' : SVC(),
            'DecisionTreeClassifier' : DecisionTreeClassifier(),
            'RandomForestClassifier' : RandomForestClassifier(),
            'AdaboostClassifier' : AdaBoostClassifier(),
            'GradientboostClassifier' : GradientBoostingClassifier()
            }

            parameters ={
            'KNeighborsClassifier' : {'n_neighbors' : [1,2,3,4,5,6,7,8,9]},

            'GaussianNB' : {},

            'SVC' : {'kernel' : ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                     'C' : [1,2,3,4,5]},

            'DecisionTreeCalssifier' : {'criterion' : ['gini', 'entropy' ,'log_loss'],
                                        'splitter' : ['best', 'random'], 
                                        'max_feature' : ['sqrt', 'log2'],
                                        'max_depth' : [1,2,3,4,5,6,7]},

            'RandomForestClassifier' : {'criterion' : ['gini', 'entropy', 'log_loss'],
                                        'max_depth' : [1,2,3,4,5,6,7],
                                        'n_estimators' : [8,16,32,64]},

            'AdaboostClassifier' : {'learning_rate' : [0.1,0.01,0.5,0.001],
                                    'n_estimators' : [8,16,32,64]},

            'GradientboostClassifier' : {'loss' : ['log_loss', 'exponential'],
                                         'learning_rate' : [0.1, 0.01,0.5,0.001],
                                         'n_estimators' : [8,16,32,64],
                                         'subsample' : [0.6,0.7,0.75,0.8,0.85,0.9],
                                         'criterion' : ['friedman_mse', 'squared_error'],
                                         'max_features' : ['sqrt' , 'log2']}
            
            }

            model_report:dict = model_evaluation(X_train, y_train, X_test, y_test, models, parameters)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            print("This is the best Model")
            print(best_model)

            model_name = list(parameters.keys())

            actual_model = ''
            for model in model_name:
                if best_model_name == model:
                    actual_model = actual_model + model

            best_parameters = parameters[actual_model]

            mlflow.set_registry_uri('https://dagshub.com/Adeshmukh-09/Market_Pulse_Analyzing_Popularity_Patterns_for_Apparel_Brands.mlflow')
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():
                predicted_qualities = best_model.predict(X_test)
                (accuracyscore, precisionscore, f1score, recallscore) = self.evaluation_metric(y_test,predicted_qualities)

                mlflow.log_params(best_parameters)
                mlflow.log_metric('accuracy_score',accuracyscore)
                mlflow.log_metric('precision_score',precisionscore)
                mlflow.log_metric('F1_score',f1score)
                mlflow.log_metric('recall_score',recallscore)
                

                if tracking_url_type_store != 'file':
                    mlflow.sklearn.log_model(best_model, 'model', registered_model_name = actual_model)

                else:
                    mlflow.sklearn.log_model(best_model, 'model')


            if best_model_score < 0.6:
                raise CustomException('No best model found')
            
            logging.info(f'Best model found')

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predict = best_model.predict(X_test)
            accuracyscore = accuracy_score(y_test,predict)

            return accuracyscore

            
            
        except Exception as e:
            raise CustomException(e,sys)
        

        
        



