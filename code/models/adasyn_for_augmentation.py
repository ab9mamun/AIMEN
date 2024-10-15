import pandas as pd
import numpy as np
def balance(ada, X, Y, feature_names, categorical_columns, master_lower_bound):
    """
    Balance the dataset using ADASYN.
    :param ada: The ADASYN object.
    :param X: The data.
    :param Y: The labels.
    :param feature_names: The feature names.
    :param categorical_columns: The categorical columns.
    :param master_lower_bound: The master lower bound.
    :return: The balanced dataset.
    """
    X_resampled, Y_resampled = ada.fit_resample(X, Y)

    #X_resampled = np.round(X_resampled).astype(int) #allowing fractional values!
    X_resampled = pd.DataFrame(X_resampled, columns=feature_names)
    X_resampled[X_resampled < master_lower_bound] = master_lower_bound
    for col in categorical_columns:
        values = X_resampled[col].clip(0, 1).to_numpy()
        X_resampled[col] = np.round(values).astype(int)
    return X_resampled.values.astype(float), Y_resampled.astype(int)
