import os
import sys
import numpy as np 
import pandas as pd
from src.Market_Pulse_Analyzing_Popularity_Patterns_for_Apparel_Brands.logger import logging 
from src.Market_Pulse_Analyzing_Popularity_Patterns_for_Apparel_Brands.exception import CustomException
import dill
import pickle 

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok = True)
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)