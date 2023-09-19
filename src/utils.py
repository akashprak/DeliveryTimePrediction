import os
import pickle
from sklearn.metrics import r2_score
from geopy.distance import geodesic
import numpy as np

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e)


def evaluate_model(X_train, X_test, y_train, y_test, models):
    try:
        report = {}
        for i in models:
            model = models[i]

            # Train model
            model.fit(X_train,y_train)

            # Predict Testing data
            y_pred =model.predict(X_test)

            # Get R2 scores for the test data
            test_model_score = r2_score(y_test,y_pred)
            report[i] =  test_model_score

        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e)


def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e)


def geo(sample):
    coordinates = ["Restaurant_latitude", "Restaurant_longitude", 
                    "Delivery_location_latitude", "Delivery_location_longitude"]
    if sample[coordinates].notnull().all():
        return geodesic(
                (sample["Restaurant_latitude"], sample["Restaurant_longitude"]), 
                (sample["Delivery_location_latitude"], sample["Delivery_location_longitude"])).km
    else:
        return np.nan    # passing as NaN if any coordinate is null