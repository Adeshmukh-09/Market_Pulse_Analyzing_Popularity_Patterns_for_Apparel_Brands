import sys
import os 
import pandas as pd
from src.Market_Pulse_Analyzing_Popularity_Patterns_for_Apparel_Brands.logger import logging
from src.Market_Pulse_Analyzing_Popularity_Patterns_for_Apparel_Brands.exception import CustomException
from src.Market_Pulse_Analyzing_Popularity_Patterns_for_Apparel_Brands.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = os.path.join('Artifacts','model.pkl')
            preprocessor_path = os.path.join('Artifacts','preprocessor.pkl')
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred 

        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self, store_ratio:float, basket_ratio:float, category_1:int, store_score:float, category_2:int, store_presence:float, score_1:float, score_2:float, score_3:float, score_4:float, time:int):
        self.store_ratio = store_ratio
        self.basket_ratio = basket_ratio
        self.category_1 = category_1
        self.store_score = store_score
        self.category_2 = category_2
        self.store_presence = store_presence 
        self.score_1 = score_1
        self.score_2 = score_2
        self.score_3 = score_3
        self.score_4 = score_4
        self.time = time 

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict={
                'store_ratio' : [self.store_ratio],
                'basket_ratio' : [self.basket_ratio],
                'category_1' : [self.category_1],
                'store_score' : [self.store_score],
                'category_2' : [self.category_2],
                'store_presence' : [self.store_presence],
                'score_1' : [self.score_1],
                'score_2' : [self.score_2],
                'score_3' : [self.score_3],
                'score_4' : [self.score_4],
                'time' : [self.time]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e,sys)

        
