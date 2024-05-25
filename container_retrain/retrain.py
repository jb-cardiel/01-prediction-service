import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, MinMaxScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from sklearn.metrics import f1_score

import dill

import argparse

# Features

MAIN_ID = "id"

NUMERICAL_FEATURES = [
    'age',
    'balance',
    'duration',
    'campaign',
    'pdays',
    'previous'
    ]

CATEGORICAL_FEATURES = [
    'job',
    'marital',
    'education',
    'default',
    'housing',
    'loan',
    'contact',
    'day_of_week',
    'month',
    'poutcome'
]

TARGET_FEATURE = 'y'

# Classes

class TargetEncodingTransformer(BaseEstimator, TransformerMixin):
    ### Uses mean of categories to encode feature the lowest the highest presence of positive category
    ### It deals with unknown categories using the maximum encoding + 1
    
    def __init__(self, categorical_features, target_feature):
        self.categorical_features = categorical_features
        self.target_feature = target_feature

    def fit(self, X, y=None):
        X_copy = X.copy()
        
        cat_dict = {}
        for cat_ in self.categorical_features:
            temp_dict = {val_:ii for ii, val_ in enumerate(X_copy.groupby(cat_)[self.target_feature].mean().sort_values(ascending=False).index)} 

            cat_dict[cat_] = temp_dict
        self.cat_dict = cat_dict
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for cat_, dict_ in self.cat_dict.items():
            X_copy[f"{cat_}_tenc"] = X_copy[cat_].map(dict_)
            ### Fill missing categories with maximum values + 1
            X_copy[f"{cat_}_tenc"] = X_copy[f"{cat_}_tenc"].fillna(max(dict_.values())+1)
        return X_copy

class OneHotEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features):
        self.categorical_features = categorical_features
        self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

    def fit(self, X, y=None):
    
        self.encoder.fit(X[self.categorical_features])
        self.encoded_features = self.encoder.get_feature_names_out()
    
        return self
    
    def transform(self, X):
        X_copy = X.copy()
       
        X_copy[[f"{f_}_ohenc" for f_ in self.encoded_features]] = self.encoder.transform(X_copy[self.categorical_features])
        
        return X_copy

class SmoothedTargetEncodingTransformer(BaseEstimator, TransformerMixin):
    ### Uses smoothed target encoding
    ### https://www.kaggle.com/code/ryanholbrook/target-encoding
    ### m is a value between 0 to 100
    
    def __init__(self, categorical_features, target_feature, m=25):
        self.categorical_features = categorical_features
        self.target_feature = target_feature
        self.m = m

    def fit(self, X, y=None):
        X_copy = X.copy()

        self.overall_average = X_copy[self.target_feature].mean()
        
        cat_dict = {}
        
        for cat_ in self.categorical_features:
            agg_df = X_copy.groupby(cat_).agg({self.target_feature:'mean', cat_:'count'})
            ### weight = counts of the category in set / counts of the category in set + m
            agg_df["weight"] = 1.0/(1.0 + self.m/agg_df[cat_])
            ### encoding = weight * in category average of target + (1 - weight) * overall average of target
            agg_df["encoding"] = agg_df["weight"]*agg_df[self.target_feature] + (1.0 - agg_df["weight"])*self.overall_average
            
            cat_dict[cat_] = agg_df["encoding"].to_dict()
            
        self.cat_dict = cat_dict
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for cat_, dict_ in self.cat_dict.items():
            X_copy[f"{cat_}_tenc"] = X_copy[cat_].map(dict_)
            ### Fill missing categories with overall average of target
            X_copy[f"{cat_}_tenc"] = X_copy[f"{cat_}_tenc"].fillna(self.overall_average)
        return X_copy

class PolynomialFeaturesTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, numerical_features, degree=2, interaction_only=False):
        self.numerical_features = numerical_features
        self.degree = degree
        self.interaction_only = interaction_only
        self.transformer = PolynomialFeatures(degree=self.degree, interaction_only=self.interaction_only, include_bias=False)

    def fit(self, X, y=None):
        X_copy = X.copy()
        
        self.transformer.fit(X_copy[self.numerical_features])
        self.polynomial_features = [f_.replace(" ", "_") for f_ in self.transformer.get_feature_names_out()]
        
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        X_copy[[f"{f_}_poly" for f_ in self.polynomial_features]] = self.transformer.transform(X_copy[self.numerical_features])
        
        return X_copy

class MinMaxScalerTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, numerical_features):
        self.numerical_features = numerical_features
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        X_copy = X.copy()
        
        ### Check for polynomial features
        poly_feats = [f_ for f_ in X_copy.columns if "_poly" in f_]
        
        if len(poly_feats) > 0:
            self.scaler.fit(X_copy[self.numerical_features+poly_feats])
        else:
            self.scaler.fit(X_copy[self.numerical_features])
        
        
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        ### Check for polynomial features
        poly_feats = [f_ for f_ in X_copy.columns if "_poly" in f_]
        
        if len(poly_feats) > 0:
            X_copy[[f"{f_}_scaled" for f_ in self.numerical_features+poly_feats]] = self.scaler.transform(X_copy[self.numerical_features+poly_feats])
        else:
            X_copy[[f"{f_}_scaled" for f_ in self.numerical_features]] = self.scaler.transform(X_copy[self.numerical_features])
        
        
        
        return X_copy

class XGBClassifierTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, main_id, target_feature, estimators=500):
        self.main_id = main_id
        self.target_feature = target_feature
        self.estimators = estimators
        
        self.classifier = XGBClassifier(n_estimators=self.estimators)

    def fit(self, X, y=None):
        X_copy = X.copy()
        
        # Get features to use
        model_features = [f_ for f_ in X_copy.columns if ("_scaled" in f_) | ("_ohenc" in f_)]
        
        self.model_features = model_features
        
        # Fit model
        self.classifier.fit(X_copy[self.model_features].values, X_copy[self.target_feature].values)
        
        # Get predictions to fit threshold optimizing f1 score
        train_pred = self.classifier.predict_proba(X_copy[model_features].values)[:,1]

        thresholds_ = np.arange(0, 1, 0.05)
        f1_scores_ = []

        for th_ in thresholds_:
            f1_score_ = f1_score(X_copy[self.target_feature].values, (train_pred > th_).astype(int))
            f1_scores_.append(f1_score_)
            
        self.best_threshold = thresholds_[np.argmax(f1_scores_)]
        
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        pred_X = self.classifier.predict_proba(X_copy[self.model_features].values)[:,1]
        binary_X = (pred_X > self.best_threshold).astype(int)
        
        predictions = pd.DataFrame(data={"id":X_copy[self.main_id].values, "prediction_float":pred_X, "prediction_boolean":binary_X}, index=X_copy.index)        
        
        return predictions

def retrain(test_mode=True):

    # TODO: Include ID for record so api can send it too

    # Load data
    dataset = pd.read_csv("data/dataset.csv")

    # Fill nan values from categorical features    
    dataset.fillna('unknown', inplace=True)

    # Change dtype of target feature
    dataset[TARGET_FEATURE] = dataset[TARGET_FEATURE].map({"yes":1, "no":0})

    train_df, _ = train_test_split(
        dataset, 
        test_size=0.2, 
        random_state=datetime.datetime.today().microsecond
        )

    # Create pipeline
    pipeline = Pipeline(
        [
            ('target_encoding', SmoothedTargetEncodingTransformer(
                categorical_features=CATEGORICAL_FEATURES, 
                target_feature=TARGET_FEATURE, 
                m=0.05)
            ),
            
            ('one_hot_encoding', OneHotEncoderTransformer(
                categorical_features=CATEGORICAL_FEATURES)
            ),
            
            ('polynomial_features', PolynomialFeaturesTransformer(
                numerical_features=NUMERICAL_FEATURES, 
                degree=2, 
                interaction_only=True)
            ),
            
            ('min_max_scaler',  MinMaxScalerTransformer(
                numerical_features=NUMERICAL_FEATURES)
            ),
            
            ('xgboost_classfier',  XGBClassifierTransformer(
                main_id=MAIN_ID,
                target_feature=TARGET_FEATURE)
            )
        ]
    )

    # Fit pipeline
    pipeline.fit(train_df)

    # Save pipeline in dill file if not in test mode
    if not(test_mode):
        with open("current_model.dill", "wb") as f_:
            dill.dump(pipeline, f_, recurse=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Test mode flag")

    parser.add_argument('--test_mode', action='store_true', default=False)
    args = parser.parse_args()

    retrain(args.test_mode)
