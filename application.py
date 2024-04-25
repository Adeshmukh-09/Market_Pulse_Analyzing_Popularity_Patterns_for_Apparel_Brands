import sys
from src.Market_Pulse_Analyzing_Popularity_Patterns_for_Apparel_Brands.logger import logging
from src.Market_Pulse_Analyzing_Popularity_Patterns_for_Apparel_Brands.exception import CustomException
from src.Market_Pulse_Analyzing_Popularity_Patterns_for_Apparel_Brands.components.data_ingestion import DataIngestion
from src.Market_Pulse_Analyzing_Popularity_Patterns_for_Apparel_Brands.components.data_transformation import DataTransformation
from src.Market_Pulse_Analyzing_Popularity_Patterns_for_Apparel_Brands.components.model_trainer import ModelTrainer
from src.Market_Pulse_Analyzing_Popularity_Patterns_for_Apparel_Brands.pipeline.prediction_pipeline import CustomData, PredictPipeline
from flask import Flask, request, render_template

application = Flask(__name__)
app = application 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    
    else:
        data = CustomData(
            store_ratio = request.form.get('store_ratio'),
            basket_ratio = request.form.get('basket_ratio'),
            category_1 = request.form.get('category_1'),
            store_score = request.form.get('store_score'),
            category_2 = request.form.get('category_2'),
            store_presence = request.form.get('store_presence'),
            score_1 = request.form.get('score_1'),
            score_2 = request.form.get('score_2'),
            score_3 = request.form.get('score_3'),
            score_4 = request.form.get('score_4'),
            time = request.form.get('time')
        )

        pred_df = data.get_data_as_dataframe()
        print(pred_df)
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(pred_df)
        result = round(pred[0])
        return render_template('home.html',final_result = result)





if __name__ == "__main__":
    logging.info("Execution has started")

    try:
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        data_transformation = DataTransformation()
        train_arr, test_arr = data_transformation.initiate_data_transformation(train_data_path,test_data_path)
        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr, test_arr))
        app.run(host = '0.0.0.0', port = 8080, debug = True )
    
    except Exception as e:
        logging.info('Exception Ocurred')
        raise CustomException(e,sys) 