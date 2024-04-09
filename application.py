import sys
from src.Market_Pulse_Analyzing_Popularity_Patterns_for_Apparel_Brands.logger import logging
from src.Market_Pulse_Analyzing_Popularity_Patterns_for_Apparel_Brands.exception import CustomException

if __name__ == "__main__":
    logging.info("Execution has started")

    try:
        a = 10/0
    
    except Exception as e:
        logging.info('Exception Ocurred')
        raise CustomException(e,sys) 