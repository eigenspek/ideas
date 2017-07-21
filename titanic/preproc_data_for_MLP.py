import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_train = pd.read_csv('~/Desktop/Programmes/python/titanic/titanic_data/train.csv')
data_test = pd.read_csv('~/Desktop/Programmes/python/titanic/titanic_data/test.csv')

def drop_feature(df):
    return df.drop(['Name','Ticket','Cabin'], axis=1)

def transform_sex(df):
    df.Sex = df.Sex.fillna('N')
    return df

def simplify_fares(df):
    df.Fare = df.Fare.astype('float')
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

def fill_missing_embarked(df):
        df.Embarked = df.Embarked.fillna('NA')
        return df
    
def simplify_ages(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df

def split_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    a = df.Cabin.str.extract('([A-Z])([0-9]+)', expand = False)
    df['CabinLevel'] = a[0]
    df['CabinNumber'] = a[1]
    return df

def transform_cabin_data(df):
    df.CabinLevel = df.CabinLevel.fillna('N')
    df.CabinNumber = df.CabinNumber.astype('float')
    df.CabinNumber = df.CabinNumber.fillna(-0.5)
    bins = (-1, 0, 90, 148)
    group_names =['Unknown','Rear','Front']
    categories = pd.cut(df.CabinNumber, bins, labels=group_names)
    df.CabinNumber = categories
    return df

def transform_data(df):
    df = simplify_fares(df)
    df = simplify_ages(df)
    df = fill_missing_embarked(df)
    df = split_cabins(df)
    df = transform_cabin_data(df)
    df = transform_sex(df)
    df = drop_feature(df)
    return df

train_set = transform_data(data_train)
test_set = transform_data(data_test)

from sklearn import preprocessing
def encode_features(df_train, df_test):
    features = ['Fare', 'Age', 'Sex', 'CabinLevel' , 'CabinNumber', 'Embarked' ]
    df_combined = pd.concat([df_train[features], df_test[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test
    
train_set, test_set = encode_features(train_set, test_set)
os.chdir(os.getcwd() + "/titanic_data")
train_set.to_csv('preproc_train.csv', index = False)
test_set.to_csv('preproc_test.csv', index = False)
