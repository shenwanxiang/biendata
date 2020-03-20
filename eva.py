import pandas as pd
import numpy as np
np.random.seed(123)

#![ax](https://biendata.com/media/competition/2019/12/31/15777543916044445SMAPE-small.png)

def smape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.nanmean(diff)