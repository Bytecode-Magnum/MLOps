import pandas as pd
from src.utils import load_model
from src.exception import CustomException
import sys

class PredictPipeline:
    def __init__(self):
        pass
    
    def make_prediction(self, df):
        try:
            model_path = './artifacts/trained_model.pkl'  # Use '/' instead of '\'
            model = load_model(model_path)
            preprocessor_path = './artifacts/preprocessor.pkl'
            preprocessor = load_model(preprocessor_path)
            scaled_df = preprocessor.transform(df)
            prediction = model.predict(scaled_df)
            return prediction
        except Exception as e:
            raise CustomException(e, sys)

