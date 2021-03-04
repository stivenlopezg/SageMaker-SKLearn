import sys
import logging

# Logger Configuration -----------------------------------------------------------------------------------------------

logger_app_name = 'SageMaker-Exercise'
logger = logging.getLogger(logger_app_name)
logger.setLevel(logging.INFO)
consoleHandle = logging.StreamHandler(sys.stdout)
consoleHandle.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
consoleHandle.setFormatter(formatter)
logger.addHandler(consoleHandle)

# AWS ----------------------------------------------------------------------------------------------------------------

AWS_REGION = 'us-east-1'

# Project ------------------------------------------------------------------------------------------------------------

target = 'median_house_value'

features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity']

numerical_features = ['housing_median_age', 'total_rooms',
                      'total_bedrooms', 'population', 'households', 'median_income']

categorical_features = ['ocean_proximity']

dummies_features = ['ocean_proximity_<1H OCEAN', 'ocean_proximity_INLAND',
                    'ocean_proximity_ISLAND', 'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN']
