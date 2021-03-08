from __future__ import print_function
import os
import json
import joblib
import argparse
import numpy as np
import pandas as pd
from io import StringIO
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
    data = pd.concat(objs=[pd.read_csv(file, names=features + [target]) for file in input_files])
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


def input_fn(input_data, content_type):
    """
    Toma los datos de entrada y los carga
    """
    if content_type == 'text/csv':
        df = pd.read_csv(StringIO(input_data), sep=',', header=None)
        if len(df.columns) == len(features) + 1:
            df.columns = features + [target]
        elif len(df.columns) == len(features):
            df.columns = features
        return df
    else:
        raise ValueError(f'{content_type} not supported by script')


def predict_fn(input_data, model):
    features = model.transform(input_data)
    if target in input_data:
        return np.insert(features, 0, input_data[target], axis=1)
    else:
        return features


def output_fn(prediction, accept):
    if accept == 'application/json':
        instances = []
        for row in prediction.tolist():
            instances.append({'features': row})

        json_output = {'instances': instances}
        return worker.Response(json.dumps(json_output), accept=accept, mimetype=accept)
    elif accept == 'text/csv':
        return worker.Response(encoders.encode(prediction, accept), accept=accept, mimetype=accept)
    else:
        raise RuntimeError(f'{accept} accept type is not supported by this script.')