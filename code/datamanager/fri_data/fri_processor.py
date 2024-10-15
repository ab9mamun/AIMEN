from . import fri_data_driver as dd
import random
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
def create_features_with_all(df, outcome):
    m, o, f, d = dd.get_risk_factors() # maternal, obstetrical, fetal, and delivery risks
    feature_variables = m + o + f + d #combine them all

    return create_features(df, outcome, feature_variables)


def create_features(df, outcome, feature_variables):
    """
    This function is used to create the features and labels for the given dataframe and outcome variable.
    :param df:
    :param outcome:
    :param feature_variables:
    :return:
    """
    df = df[feature_variables + [outcome]]
    print(f"Shape of the dataframe before dropping for NA, {df.shape}")
    df = df.dropna()
    outcome_arr = df[outcome].to_numpy()
    outcome_arr = outcome_arr.astype(int)
    print(f"Shape of the dataframe after dropping for NA, {df.shape}")
    features = df[feature_variables].values
    print(f"Shapes of the feature array: {features.shape}, and outcome array: {outcome_arr.shape}")
    print("Returning only part of the data for testing purposes")
    features, labels = shuffle(features, outcome_arr, random_state=0)
    return features, labels, feature_variables  # [:len(features) // 5 * 4], outcome_arr[:len(outcome_arr) // 5 * 4]
    # return features[:len(features)//5*4], outcome_arr[:len(outcome_arr)//5*4]
def create_features_except_heartrate(df, outcome):
    m, o, f, d = dd.get_risk_factors() # maternal, obstetrical, fetal, and delivery risks
    feature_variables = m + o + f+ d[5:] #combine them all except for the heart rate variables (indices 0 to 4 of delivery risks)

    return create_features(df, outcome, feature_variables)


def get_categorical_columns():
    return dd.get_categorical_columns()