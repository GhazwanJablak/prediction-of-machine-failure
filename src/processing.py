import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict
import sklearn
import numpy as np


def null_association(
        df:pd.DataFrame
        ) -> pd.DataFrame:
    """
    create a contingency table between features with null values.

    Parameters:
    df: Dataframe containing all features.

    Returns:
    association_table: contingency dataframe of null values.
    """
    temp = df.isnull().sum()
    cols = list(temp[temp>0].index)
    associations_dict={}
    for col in cols:
        associations_dict[col] = df[df[col].isnull()][cols].isnull().sum().values
    associations_table = pd.DataFrame(associations_dict, index = [cols])
    return associations_table

class MedianImputer(
        ) :
    """extract the median value of the varible per group of Type categorical variable and use it to imput nulls.

    Parameters:
    features: List of features.

    Returns:
    df: imputed dataframe.
    """
    def __init__(self, features):
        self.features = features
        self.col_dict = {}
    def fit(self, df):
        temp = df.isnull().sum()
        self.cols = list(temp[temp>0].index)
        for col in self.cols:
            self.col_dict[col] = df.groupby("Type")[col].transform("median")
        return self
    def transform(self, df):
        for col in self.cols:
            df[col] = df[col].fillna(self.col_dict[col])
        return df



def data_dict(
        dataframe:pd.DataFrame, 
        column_name:str
        ) -> Dict:
    """
    Get value of counts of a numeric variables available in unique categories of Machine failure variable 

    Parameters:
    dataframe: Dataframe containing all features.
    column_name: name of the numeric feature.

    Returns:
    column_dict: Dictionary of count per category.
    """
    t = dataframe[["Machine failure", column_name]]\
        .value_counts().reset_index().sort_values(by=[column_name, "Machine failure"])
    column_dict = {k: [list(t[t[column_name] == k]["Machine failure"]),
                       list(t[t[column_name] == k]["count"])]
                   for k in t[column_name].unique()}
    return column_dict


class CountEncoder():
    """
    Class to encode categorical variables using count of unique values.

    Parameters:
    cat_variables: list of categorical variables.

    Returns:
    df: Dataframe with encoded variables.
    """
    def __init__(self, cat_variables):
        self.cat_variables = cat_variables
        self.col_dict = {}
    def fit(self, dataframe):
        for col in self.cat_variables:
            self.col_dict[col] = dataframe[col].value_counts().to_dict()
        return self
    def transform(self, dataframe):
        for col in self.cat_variables:
            dataframe[col] = dataframe[col].map(self.col_dict[col])
        return dataframe
    

def classifier_pipeline(
        df:pd.DataFrame, 
        target_col:str, 
        num_cols:List, 
        cat_cols:List, 
        model:sklearn.base.ClassifierMixin, 
        test_size:float=0.2, 
        random_state:int=42
        ) -> Tuple[np.ndarray, np.ndarray]:
    """Apply machine learning pipeline on training, the pipeline, splits the data, encode categorical variables
     and fit a classifier to training data and estimate prediction on test data

     Parameters:
     df: Dataframe of training data.
     target_col: name of dependant variable.
     num_cols: list of numeric features.
     cat_cols: list of categorical features.
     model: scikit-learn classifier.
     test_size: proportion of data to keep for testing.
     random_state: 
     """
    target = df[target_col]
    features_all = df[num_cols+cat_cols]
    X_train, X_test, y_train, y_test = train_test_split(features_all, target, test_size=test_size, stratify=target, random_state=random_state)
    imputer = MedianImputer(num_cols+cat_cols)
    imputer.fit(X_train)
    X_train_imp = imputer.transform(X_train)
    X_test_imp = imputer.transform(X_test)
    encoder = CountEncoder(cat_variables=cat_cols)
    encoder.fit(X_train_imp)
    X_train_en = encoder.transform(X_train_imp)
    X_test_en = encoder.transform(X_test_imp)
    model.fit(X_train_en, y_train)
    predictions = model.predict(X_test_en)
    return y_test, predictions
