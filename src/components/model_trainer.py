from src.exception import CustomException
from src.logger import logging
import os
import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost  as xgb
from src.utils import load_model
from dataclasses import dataclass
from sklearn.metrics import precision_score,accuracy_score,classification_report
from src.utils import load_model,save_object
from src.utils import evaluate_model
from sklearn.ensemble import GradientBoostingClassifier


@dataclass
class ModelTrainerConfig:
  trained_model_file_path=os.path.join('artifacts','trained_model.pkl')

class ModelTrainer:
  def __init__(self):
    self.model_trainer_config=ModelTrainerConfig()

  def initiate_model_training(self,train_array,test_array):
    report=[]
    eval={}
    try:
     #" this is first model on testing ie random forest classifier"
      logging.info('data splitting initiated for model traning')
      X_train,y_train,X_test,y_test=train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1]
      rf=RandomForestClassifier()
      precision,accuracy,classification=evaluate_model(X_train,y_train,X_test,y_test,rf)
      print('Random Forest')
      print(f"Precision Score {precision}\n Accuracy Score {accuracy}\n Classification Report {classification}")
      eval['model']='RandomForest'
      eval['precision']=precision
      eval['accuracy']=accuracy
      report.append(eval)
      eval={}
      

    #: trying gradient boosting 
      logging.info('trying graident boosting for model training..........')
      
      gb=GradientBoostingClassifier()
      print('Gradient Boost')
      precision,accuracy,classification=evaluate_model(X_train,y_train,X_test,y_test,gb)
      print(f"Precision Score {precision}\n Accuracy Score {accuracy}\n Classification Report {classification}")
      eval['model']='Gradient'
      eval['precision']=precision
      eval['accuracy']=accuracy
      report.append(eval)

      eval_df=pd.DataFrame.from_records(report)
      print(eval_df)
      eval_df=eval_df.sort_values(by='accuracy')
      best_model=eval_df.loc[0]
      if best_model['model']=='RandomForest':
        save_object(
          file_path=self.model_trainer_config.trained_model_file_path,
          object=rf
        )
      if best_model['model']=='Gradient':
        save_object(
          file_path=self.model_trainer_config.trained_model_file_path,
          object=gb
        )
      print(f"{best_model['model']} IS THE BEST MODEL WITH ACCURACY OF {best_model['accuracy']}")



      

    except Exception as e:
      raise CustomException(e,sys)