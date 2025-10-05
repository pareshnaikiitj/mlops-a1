import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib

def load_data():
    # load Boston dataset manually (as instructed)
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    feature_names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target
    return df

def get_X_y(df, target_col='MEDV'):
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def preprocess(X_train, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler

def train_model(estimator, X_train, y_train):
    estimator.fit(X_train, y_train)
    return estimator

def evaluate_model(estimator, X_test, y_test):
    preds = estimator.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return mse, preds

def save_model(estimator, path):
    joblib.dump(estimator, path)
