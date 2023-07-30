import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object



@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_features = ['writing score','reading score']
            categorical_features = ['gender','race/ethnicity','parental level of education','lunch','test preparation course']

            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scalar',StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )
            
            logging.info(f'Numerical columns: {numerical_features}')
            logging.info(f'Categorical columns encoding completed: {categorical_features}')


            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_features),
                    ('cat_pipeline', cat_pipeline,categorical_features)

                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def intiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read Train and Test data completed')
            logging.info('obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformer_object()

            target_feature = 'math score'
            numerical_features = ['writing score','reading score']

            input_feature_train_df = train_df.drop(columns=[target_feature],axis=1)
            target_feature_train_df = train_df[target_feature]

            input_feature_test_df = test_df.drop(columns=[target_feature],axis=1)
            target_feature_test_df = test_df[target_feature]

            logging.info(f'Applying preprocessing object on training dataframe and testing dataframe.')

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info(f'Saced preprocessing object')

            save_object(
                file_path=self.data_transformation.preprocessor_ob_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation.preprocessor_ob_file_path,
            )
        
        except Exception as e:
            raise CustomException(e, sys)
