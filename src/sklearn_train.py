from __future__ import print_function
import os
import joblib
import argparse
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import GradientBoostingRegressor
from pipeline.custom_pipeline import ColumnsSelector, GetDummies, GetDataFrame
from config.config import numerical_features, categorical_features, dummies_features, target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    args = parser.parse_args()
    # Load data
    input_files = [os.path.join(args.train, file) for file in os.listdir(args.train)]
    data = pd.concat(objs=[pd.read_csv(file) for file in input_files])
    label = data.pop(target)
    # Transformers
    numeric_preprocessing = Pipeline(steps=[('numeric_selector', ColumnsSelector(columns=numerical_features)),
                                            ('scaler', RobustScaler()),
                                            ('numeric_df', GetDataFrame(columns=numerical_features))])

    categorical_preprocessing = Pipeline(steps=[('categoric_selector', ColumnsSelector(columns=categorical_features)),
                                                ('ohe', GetDummies(columns=categorical_features))])

    preprocessing = Pipeline(steps=[
        ('feature_union', FeatureUnion(transformer_list=[
            ('numeric', numeric_preprocessing),
            ('categoric', categorical_preprocessing)])),
        ('dataframe', GetDataFrame(columns=numerical_features + dummies_features))])
    # Model
    model = Pipeline(steps=[('preprocessing', preprocessing),
                            ('gradient_boosting', GradientBoostingRegressor())])

    model.fit(data, label)
    joblib.dump(model, filename=os.path.join(args.model_dir, 'model.joblib'))
    print('The model has been saved!')
    return True


def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model
