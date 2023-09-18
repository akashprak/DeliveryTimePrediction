import numpy as np 
import pandas as pd
import os
from dataclasses import dataclass
from geopy.distance import geodesic

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer,StandardScaler,TargetEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

# Configurations for data transformation
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')


# Class for the functions to transform coordinates into distance, transform the time feature
# and return the feature names
class _timeDist_transform:
    coordinates = ["Restaurant_latitude", "Restaurant_longitude", 
                    "Delivery_location_latitude", "Delivery_location_longitude"]
    
    def _geo(self, sample):
        if sample.notnull().all():
            return geodesic(
                    (sample["Restaurant_latitude"], sample["Restaurant_longitude"]), 
                    (sample["Delivery_location_latitude"], sample["Delivery_location_longitude"])).km
        else:
            return np.nan    # passing as NaN if any coordinate is null
 
    def timeDist(self, df: pd.DataFrame):
        """function to convert coordinates to distance in kilometres"""
        try:
            logging.info('Converting coordinates into distance')
            df["Distance"] = df.apply(self._geo, axis=1) 
            df = df.drop(columns=self.coordinates)

            logging.info("Converting feature to datetime format and extract 'hour' part")
            df['Time_Order_picked'] = pd.to_datetime(df['Time_Order_picked'], errors='coerce', format='%H:%M').dt.hour
            return df
        except Exception as e:
            logging.exception('Error during time-distance conversion')

    def feature_names(self, _, feature_names):
        features  = [i for i in feature_names if i not in self.coordinates]
        features.append('Distance')
        return features


# Deploys the pipeline and data transformation
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        self.timeDist = _timeDist_transform()

    def mapper(self, df :pd.DataFrame):
        '''mapping function for the categorical columns'''

        # Defining the custom ranking for each ordinal variable
        Weather_conditions_map={'Fog':3,'Cloudy':3,'Stormy':2,'Sandstorms':2,'Windy':2,'Sunny':1}
        Road_traffic_density_map={'Jam':4, 'High':3, 'Medium':2, 'Low':1}
        Type_of_vehicle_map={'motorcycle':2, 'bicycle':2, 'scooter':1, 'electric_scooter':1}
        Festival_map={'Yes':1, 'No':0}
        City_map={'Semi-Urban':3, 'Metropolitian':2, 'Urban':1}

        df['Weather_conditions'] = df['Weather_conditions'].map(Weather_conditions_map)
        df['Road_traffic_density'] = df['Road_traffic_density'].map(Road_traffic_density_map)
        df['Type_of_vehicle'] = df['Type_of_vehicle'].map(Type_of_vehicle_map)
        df['Festival'] = df['Festival'].map(Festival_map)
        df['City'] = df['City'].map(City_map)
        return df

    def get_data_transformation_object(self):   
        try:
            logging.info('Data Transformation initiated')

            # Defining the columns to apply different transformations
            numerical_columns = ['Delivery_person_Age', 'Delivery_person_Ratings', "Restaurant_latitude", "Restaurant_longitude",
                                 "Delivery_location_latitude", "Delivery_location_longitude", 'Time_Order_picked']
            categorical_columns = ['Weather_conditions', 'Road_traffic_density', 'Vehicle_condition', 'Type_of_vehicle',
                                   'multiple_deliveries', 'Festival', 'City']
            id_column = ['Delivery_person_ID']
            
            logging.info('Pipeline Initiated')

            # Delivery person ID pipeline
            id_pipeline = Pipeline(
                steps=[
                    ('target_encoder_id', TargetEncoder(target_type='continuous')),
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ])

            # Numerical pipeline
            numerical_pipeline = Pipeline(
                steps=[
                    ('time_distance', FunctionTransformer(self.timeDist.timeDist,
                                                        feature_names_out = self.timeDist.feature_names)),
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ])

            # categorical pipeline
            categorical_pipeline = Pipeline(
                steps=[
                    ('mapper', FunctionTransformer(self.mapper, feature_names_out='one-to-one')),
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('scaler', StandardScaler())
                ])

            preprocessor = ColumnTransformer([
                ('id', id_pipeline, id_column),
                ('num_pipe', numerical_pipeline, numerical_columns),
                ('cat_pipe', categorical_pipeline, categorical_columns)
            ])
            
            logging.info('Pipeline Completed')
            return preprocessor

        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e)
        
    def initiate_data_transformation(self,train_path, test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')
            preprocessing_obj = self.get_data_transformation_object()

            # target feature and redundant features to be dropped 
            target_feature = 'Time_taken (min)'
            drop_features = ['ID', 'Time_Orderd', 'Order_Date', 'Type_of_order', target_feature]

            input_feature_train_df = train_df.drop(columns=drop_features)
            target_feature_train_df = train_df[target_feature]

            input_feature_test_df = test_df.drop(columns=drop_features)
            target_feature_test_df = test_df[target_feature]
            
            # Data transformation using preprocessor
            logging.info("Applying preprocessor object on training and testing datasets.")
            X_train_arr = preprocessing_obj.fit_transform(input_feature_train_df, y=target_feature_train_df)
            X_test_arr = preprocessing_obj.transform(input_feature_test_df)

            y_train_arr = np.array(target_feature_train_df)
            y_test_arr = np.array(target_feature_test_df)

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
                )
            logging.info('Preprocessor pickle file saved')

            return (
                X_train_arr,
                X_test_arr,
                y_train_arr,
                y_test_arr
            )
            
        except Exception as e:
            logging.info("Exception occured in the function: initiate_data_transformation")
            raise CustomException(e)