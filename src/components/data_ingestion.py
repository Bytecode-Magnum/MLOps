from src.exception import CustomException
from src.logger import logging
import os
from dataclasses import dataclass
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from src.components.data_transformation import Data_transformation
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionconfig:
  train_data_path=os.path.join("artifacts",'train.csv')
  test_data_path=os.path.join('artifacts','test.csv')
  raw_data_path=os.path.join('artifacts','raw.csv')


class DataIngestion:
  def __init__(self):
    self.ingestion_config=DataIngestionconfig()


  def initiate_data_ingestion(self):
    logging.info('initiate data ingestion')
    try:
      df=pd.read_csv('./notebook\dataset\loan_approval_prediction_dataset.csv')
      os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)   #created a folder of name artifacts"
      df.to_csv(self.ingestion_config.raw_data_path,index=False)                          #: created a csv of name raw.csv
      train_set,test_set=train_test_split(df,test_size=.2,random_state=43)
      train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
      test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

      return(
        self.ingestion_config.train_data_path,
        self.ingestion_config.test_data_path
      )

    except Exception as e:
      raise CustomException(e,sys)
    
if __name__=="__main__":
  obj=DataIngestion()
  train_data_path,test_data_path=obj.initiate_data_ingestion()
  data_transformation=Data_transformation()
  train_arr,test_arr,preprocessor_file_path=data_transformation.initiate_data_transformation(train_data_path,test_data_path)
  model_trainer=ModelTrainer()
  model_trainer.initiate_model_training(train_arr,test_arr)
    