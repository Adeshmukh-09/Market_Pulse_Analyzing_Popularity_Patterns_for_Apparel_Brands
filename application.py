import sys
from src.Market_Pulse_Analyzing_Popularity_Patterns_for_Apparel_Brands.logger import logging
from src.Market_Pulse_Analyzing_Popularity_Patterns_for_Apparel_Brands.exception import CustomException
from src.Market_Pulse_Analyzing_Popularity_Patterns_for_Apparel_Brands.components.data_ingestion import DataIngestion

if __name__ == "__main__":
    logging.info("Execution has started")

    try:
        data_ingestion = DataIngestion()
        train_data, test_data = data_ingestion.initiate_data_ingestion()
    
    except Exception as e:
        logging.info('Exception Ocurred')
        raise CustomException(e,sys) 