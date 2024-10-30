import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

import rampwf as rw

problem_title = "Temperature prediction as regression problem"

_event_label_names = [
"Country",
"Latitude",
"Longitude",
"dt
]

Predictions = rw.prediction_types.make_regression(label_names=_event_label)
workflow = rw.workflows.Regressor()

score_types = [
    rw.score_types.RMSE(
        name="Root-mean-square-error", precision=3, adjusted=False
    )
    
]

def _get_data(path=".", split="train"):
    # Load data from csv files into pd.DataFrame

    data_df = pd.read_csv(os.path.join(path, "data", split + ".csv"))
    data_df = data_df.dropna()
    
    #Here we drop the City column because we consider that the coordinates
    #We also drop the 'AverageTemperatureUncertainty' feature because of its closeness
    data_df=data_df.drop(columns=['City','AverageTemperatureUncertainty'])
    
    # Fit and transform the 'Country' column
    data_df['Country'] = label_encoder.fit_transform(data_df['Country'])
    
    # formatage
    Years = [np.array(data_df)[k,0].year for k in range(len(np.array(data_df)))]
    Month = [np.array(data_df)[k,0].month for k in range(len(np.array(data_df)))]
    
    data_df['Year'] = Years
    data_df['Month'] = Month
    data_df['Latitude'] = data_df['Latitude'].str[:-1]
    data_df['Latitude'] = data_df['Latitude'].astype(float)

    data_df['Longitude'] = data_df['Longitude'].str[:-1]
    data_df['Longitude'] = data_df['Longitude'].astype(float)

    # features
    X = data_df.drop(columns=['AverageTemperature','dt'])
    
    # labels
    y = ByMajorCity_true['AverageTemperature']

    return X, y
    
    
def get_test_data(path="."):
    return _get_data(path, "test")


def get_cv(X, y):
    cv = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=2)
    return cv.split(X, y, groups)
