import os
import sys
import pandas as pd 
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.Market_Pulse_Analyzing_Popularity_Patterns_for_Apparel_Brands.logger import logging 
from src.Market_Pulse_Analyzing_Popularity_Patterns_for_Apparel_Brands.exception import CustomException

@dataclass

class DataIngestionConfig:
    raw_data_path:str =os.path.join('Artifacts','raw.csv')
    train_data_path:str = os.path.join('Artifacts','train_data.csv')
    test_data_path:str = os.path.join('Artifacts','test_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info('Reading the raw data from the CSV file')

            data = pd.read_csv('D:\\Data Science Roadmap and Practice\\Machine Learning Projects\\Market_Pulse_Analyzing_Popularity_Patterns_for_Apparel_Brands\\notebook\\Data\\store_data_updated.csv')

            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)),exist_ok = True)
            data.to_csv(self.ingestion_config.raw_data_path,index = False,header = True)
            logging.info('Raw data file has been created')

            logging.info('splitting the data into train and test data using train test split')
            train_data, test_data = train_test_split( data, test_size=0.20, random_state=42)

            train_data.to_csv(self.ingestion_config.train_data_path,index = False, header = True)
            logging.info('train data file has been created')

            test_data.to_csv(self.ingestion_config.test_data_path,index = False, header = True)
            logging.info('test data file has been created')

            logging.info('Data ingestion completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as e:
            raise CustomException(e,sys)

