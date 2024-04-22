import sys
from src.Market_Pulse_Analyzing_Popularity_Patterns_for_Apparel_Brands.logger import logging
from src.Market_Pulse_Analyzing_Popularity_Patterns_for_Apparel_Brands.exception import CustomException
from src.Market_Pulse_Analyzing_Popularity_Patterns_for_Apparel_Brands.components.data_ingestion import DataIngestion
from src.Market_Pulse_Analyzing_Popularity_Patterns_for_Apparel_Brands.components.data_transformation import DataTransformation
from src.Market_Pulse_Analyzing_Popularity_Patterns_for_Apparel_Brands.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    logging.info("Execution has started")

    try:
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        data_transformation = DataTransformation()
        train_arr, test_arr = data_transformation.initiate_data_transformation(train_data_path,test_data_path)
        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr, test_arr))
    
    except Exception as e:
        logging.info('Exception Ocurred')
        raise CustomException(e,sys) 