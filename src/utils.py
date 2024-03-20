import pandas as pd
import numpy as np
import pickle
import os
import sys

from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from sklearn.metrics import precision_score,accuracy_score,classification_report

def save_object(file_path,object):
  try:
    dir_path=os.path.dirname(file_path)
    os.makedirs(dir_path,exist_ok=True)
    with open(file_path,'wb') as file_object:
      pickle.dump(object,file_object)


  except Exception as e:
    raise CustomException(e,sys)
  

def load_model(file_path):
  try:
      with open(file_path,'rb') as file_object:
        return pickle.load(file_object)
  except Exception as e:
      raise CustomException(e,sys)
  


def evaluate_model(X_train,y_train,X_test,y_test,model):
  try:
      model.fit(X_train,y_train)
      predicted_y=model.predict(X_test)
      precision,accuracy,classification=precision_score(predicted_y,y_test),accuracy_score(predicted_y,y_test),classification_report(predicted_y,y_test)
      return precision,accuracy,classification
   
  except Exception as e:
     raise CustomException(e,sys)
     
     