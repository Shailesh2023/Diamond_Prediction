import os
import sys
from src.exception import CustomException
from src.logging import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation 

# Initialize the data ingestion configuration


@dataclass
class DataIngestionconfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path :str = os.path.join('artifacts','test.csv')
    raw_data_path :str = os.path.join('artifacts','raw.csv')

# Create a class for Data Ingestion
class DataIngestion:
    def __init__(self):
        self.ingestionconfig = DataIngestionconfig()
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Starts")
        try:
            df = pd.read_csv(os.path.join("notebooks/data","gemstone.csv"))
            logging.info("Data read as Pandas Dataframe")
            os.makedirs(os.path.dirname(self.ingestionconfig.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestionconfig.raw_data_path,index = False)
            logging.info("Train test split")
            train_set,test_set=train_test_split(df,test_size=0.30)

            train_set.to_csv(self.ingestionconfig.train_data_path,index = False,header = True)
            test_set.to_csv(self.ingestionconfig.test_data_path,index = False,header = True)

            logging.info("Data Ingestion is completed")

            return(
                self.ingestionconfig.train_data_path,
                self.ingestionconfig.test_data_path

            )

        except Exception as e:
            logging.info("Exception occured at Data Ingestion Stage")
            raise CustomException(e,sys)
    
# Run data_ingeation
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path,test_data_path = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data_path,test_data_path)
    

