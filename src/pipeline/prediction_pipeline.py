import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object, geo

class CustomData:
    _numeric_features = ['Delivery_person_Age', 'Delivery_person_Ratings', "Restaurant_latitude", "Restaurant_longitude",
                          "Delivery_location_latitude", "Delivery_location_longitude", "Vehicle_condition", "multiple_deliveries"]
    
    def __init__(self, input_dict) -> None:
        self._input_dict = dict(input_dict)

    def as_dataframe(self):
        """to convert numeric featues from string to float
        and return input as dataframe"""
        for feature in self._numeric_features:
            self._input_dict[feature]=float(self._input_dict.get(feature))
        
        df = pd.DataFrame(self._input_dict, index=[0])
        return df
        

class PredictPipeline:

    def __init__(self) -> None:
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model_path = os.path.join('artifacts','model.pkl')

            self.preprocessor = load_object(preprocessor_path)
            self.model = load_object(model_path)
    
    def predict(self, dataframe):
        try:
            data_scaled = self.preprocessor.transform(dataframe)
            pred = self.model.predict(data_scaled)
            distance = geo(dataframe.loc[0])
            return pred[0], distance
            
        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e)