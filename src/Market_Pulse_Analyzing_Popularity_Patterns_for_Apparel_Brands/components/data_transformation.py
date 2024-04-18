import sys 
import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from src.Market_Pulse_Analyzing_Popularity_Patterns_for_Apparel_Brands.logger import logging 
from src.Market_Pulse_Analyzing_Popularity_Patterns_for_Apparel_Brands.exception import CustomException
from src.Market_Pulse_Analyzing_Popularity_Patterns_for_Apparel_Brands.utils import save_object 

@dataclass

class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('Artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            logging.info('Data transformation initiated')

            numerical_columns = ['store_ratio','basket_ratio','category_1','store_score','category_2','store_presence','score_1','score_2','score_3','score_4','time']

            numerical_pipeline = Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('standardscaler',StandardScaler())
            ])

            logging.info(f'numerical columns are:{numerical_columns}')

            preprocessor = ColumnTransformer(
                transformers=[('numerical_pipeline',numerical_pipeline,numerical_columns)]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('reading the test and train data')
            preprocessor_obj = self.get_data_transformer_object()

            target_column = 'popularity'

            input_feature_train_df = train_df.drop(columns = [target_column],axis = 1)
            target_feature_train_df = train_df.loc[:,[target_column]]

            input_feature_test_df = test_df.drop(columns = [target_column], axis = 1)
            target_feature_test_df = test_df.loc[:,[target_column]]

            logging.info('Applying preprocessor on the train and test data')
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info('Saving preprocessor file')

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            return(
                train_arr,
                test_arr
            )

        except Exception as e:
            raise CustomException(e,sys)

