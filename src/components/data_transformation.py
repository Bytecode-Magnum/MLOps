from src.exception import CustomException
from src.logger import logging

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer

from dataclasses import dataclass
import os
import sys
from src.utils import save_object
import pandas as pd
import numpy as np

@dataclass
class DataTransformationConfig:
  preprocessor_obj_file=os.path.join('artifacts','preprocessor.pkl')

class Data_transformation:
  def __init__(self):
    self.data_transformation=DataTransformationConfig()


  def get_datatransformation_object(self):
    try: 

        # List of numerical columns
        numerical_cols=['no_of_dependents', 'income_annum', 'loan_amount',
                        'loan_term', 'cibil_score', 'residential_assets_value',
                        'commercial_assets_value', 'luxury_assets_value',
                        'bank_asset_value']

        # List of categorical columns
        categorical_cols=['education', 'self_employed']

        # Pipeline for applying the standard scaler to the numerical columns
        num_pipeline = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]
        )

        # Pipeline for applying one hot encoding to the categorical columns
        cat_pipeline = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder())
            ]
        )

        # ColumnTransformer to bind both the pipelines
        preprocessor = ColumnTransformer(
            transformers=[
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols)
            ]
        )

        return preprocessor
    
    except Exception as e:
        raise CustomException(e,sys)

  
  def initiate_data_transformation(self,train_path,test_path):
    try:
      train_df=pd.read_csv(train_path)
      test_df=pd.read_csv(test_path)
      train_df=train_df.drop(columns=['loan_id'],axis=1)
      test_df=test_df.drop(columns=['loan_id'],axis=1)
      train_df['loan_status']=train_df['loan_status'].map({
        ' Approved':1,
        ' Rejected':0
      })
      test_df['loan_status']=test_df['loan_status'].map({
        ' Approved':1,
        ' Rejected':0
      })
      target_columns='loan_status'

      logging.info('Reading train and test data completed')
      logging.info('Obtaining preprocessing object')
      print(train_df.columns)
      preprocessor_object=self.get_datatransformation_object()
      
      input_feature_train_df=train_df.drop(columns=[target_columns],axis=1)
      target_feature_train_df=train_df[target_columns]

      input_feature_test_df=test_df.drop(columns=[target_columns],axis=1)
      target_feature_test_df=test_df[target_columns]

      logging.info('applying data preprocessing on the test and train data')

      input_feature_train_arr=preprocessor_object.fit_transform(input_feature_train_df)

      input_feature_test_arr=preprocessor_object.fit_transform(input_feature_test_df)

      logging.info('data transformation completed successfully')

      train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
      test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

      save_object(
        file_path=self.data_transformation.preprocessor_obj_file,
        object=preprocessor_object)
      logging.info('saved the preprocessor model after transformation')
      return(
        train_arr,test_arr,
        self.data_transformation.preprocessor_obj_file
      )

    except Exception as e:
      raise CustomException(e,sys)


