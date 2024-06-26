import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
project_name = "Market_Pulse_Analyzing_Popularity_Patterns_for_Apparel_Brands"

list_of_path=[
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/data_monitoring.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/training_pipeline.py",
    f"src/{project_name}/pipeline/prediction_pipeline.py",
    f"src/{project_name}/exception.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/utils.py",
    "main.py",
    "application.py",
    "requirements.txt",
    "setup.py"
]

for filepath in list_of_path:
    filepath = Path(filepath)

    filedir, filename = os.path.split(filepath)
    if filedir !="":
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"Creating Directory:{filedir} for the file {filename}")
    
    if ((not os.path.exists(filepath)) or (os.path.getsize(filepath)==0)):
        with open(filepath,'w') as f:
            pass
            logging.info(f"Creating an empty file:{filepath}")

    else:
        logging.info(f"{filename} already exist")