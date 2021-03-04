from __future__ import print_function
import os
import joblib
import argparse
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from pipeline.custom_pipeline import ColumnsSelector, GetDummies, GetDataFrame
from config.config import numerical_features, categorical_features, target, features

from sagemaker_containers.beta.framework import encoders, worker


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    args = parser.parse_args()
    # Load data
    input_files = [os.path.join(args.train, file) for file in os.listdir(args.train)]
    data = pd.concat(objs=[pd.read_csv(file) for file in input_files])
    label = data.pop(target)
    # Transformers
    numeric_preprocessing = Pipeline(steps=[('numeric_selector', ColumnsSelector(columns=numerical_features)),
                                            ('imputer', SimpleImputer(strategy='median')),
                                            ('scaler', RobustScaler()),
                                            ('numeric_df', GetDataFrame(columns=numerical_features))])

    categorical_preprocessing = Pipeline(steps=[('categoric_selector', ColumnsSelector(columns=categorical_features)),
                                                ('ohe', GetDummies(columns=categorical_features))])

    preprocessing = Pipeline(steps=[
        ('feature_union', FeatureUnion(transformer_list=[
            ('numeric', numeric_preprocessing),
            ('categoric', categorical_preprocessing)]))])

    preprocessing.fit(data)
    joblib.dump(preprocessing, filename=os.path.join(args.model_dir, 'preprocessor.joblib'))
    print('The preprocessor has been saved!')


def model_fn(model_dir):
    """
    Funcion para cargar el modelo serializado
    :param model_dir
    """
    model = joblib.load(os.path.join(model_dir, "preprocessor.joblib"))
    return model