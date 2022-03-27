import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
import joblib


def Sqrt(x):
    y = np.sqrt(x)
    return y


def Cbrt(x):
    y = np.cbrt(x)
    return y


def Log(x):
    y = np.log(1 + x)
    return y

def GetHour(x):
    y = x.hour
    z = x.minute / 60
    return y + z

def GetMonth(x):
    y = x.month
    return y

def get_feature_names(ct):
    output_features = []
    for name, pipe, features in ct.transformers_:
        for i in pipe:
            trans_features = []
            if hasattr(i,'categories_'):
                trans_features.extend(i.get_feature_names(features))
            else:
                trans_features = features
        output_features.extend(trans_features)
    return output_features





def DataProcessing(train, test, pop):
    #DATA CLEANUP

    drop_cols = [
        'Local_Authority_(District)',
        'Local_Authority_(Highway)',
        '1st_Road_Number',
        '2nd_Road_Number',
        'country',
        'Accident_ID',
        '2nd_Road_Class'
        ]

    train = train.drop(columns=drop_cols)
    test = test.drop(columns=drop_cols)


    #POPULATION CLEANUP
    pop = pop.rename(columns={
        'postcode': 'postcode_merge',
        'Variable: All usual residents; measures: Value': 'Population',
        'Variable: Males; measures: Value': 'Male_Count',
        'Variable: Females; measures: Value': 'Female_Count',
        'Variable: Lives in a household; measures: Value': 'Lives_in_Household',
        'Variable: Lives in a communal establishment; measures: Value': 'Lives_in_Communal',
        'Variable: Schoolchild or full-time student aged 4 and over at their non term-time address; measures: Value': 'Children_Count',
        'Variable: Area (Hectares); measures: Value': 'Area',
        'Variable: Density (number of persons per hectare); measures: Value': 'Pop_Density'
    }).drop(columns=['Rural Urban'])

    pop['Male_Perc'] = pop['Male_Count'] / pop['Population']
    pop['Female_Perc'] = pop['Female_Count'] / pop['Population']
    pop['Lives_in_Household_Perc'] = pop['Lives_in_Household'] / pop['Population']
    pop['Lives_in_Communal_Perc'] = pop['Lives_in_Communal'] / pop['Population']
    pop['Children_Perc'] = pop['Children_Count'] / pop['Population']

    #Train Merge
    train['postcode_merge'] = train['postcode'].str.replace(' ', '')
    train['postcode_merge'] = train['postcode_merge'].str[:-2]
    pop['postcode_merge'] = pop['postcode_merge'].str.replace(' ', '')
    train = pd.merge(train, pop, on='postcode_merge', how='left')
    train = train.drop(columns=['postcode_merge'])

    #Test Merge
    test['postcode_merge'] = test['postcode'].str.replace(' ', '')
    test['postcode_merge'] = test['postcode_merge'].str[:-2]
    pop['postcode_merge'] = pop['postcode_merge'].str.replace(' ', '')
    test = pd.merge(test, pop, on='postcode_merge', how='left')
    test = test.drop(columns=['postcode_merge'])

    cat_cols = [
    'Day_of_Week',
    #'1st_Road_Number',
    '1st_Road_Class',
    'Road_Type',
    #'2nd_Road_Class',
    'Pedestrian_Crossing-Human_Control',
    'Pedestrian_Crossing-Physical_Facilities',
    'Light_Conditions',
    'Weather_Conditions',
    'Road_Surface_Conditions',
    'Special_Conditions_at_Site',
    'Carriageway_Hazards',
    'state',
    #'Local_Authority_(District)',
    #'Local_Authority_(Highway)',
    'Month'
    ]

    num_cols = [
        'Police_Force',
        'Number_of_Vehicles',
        'Speed_limit',
        #'Lives_in_Household_Perc',
        #'Lives_in_Communal_Perc',
        'Hour'
    ]

    bin_cols = [
        'Urban_or_Rural_Area',
        'Did_Police_Officer_Attend_Scene_of_Accident'
    ]

    sqrt_cols = [
        'Children_Count',
        'Male_Count',
        #'Female_Count',
        'Population',
        'Lives_in_Household'
    ]

    cbrt_cols = [
        'Children_Perc',
        'Male_Perc',
        #'Female_Perc'
    ]

    log_cols = [
        'Lives_in_Communal',
        'Area',
        'Pop_Density',
    ]

    target_col = ['Number_of_Casualties']

    #Train Times
    train['Time'] = pd.to_datetime(train['Time'])
    train['Hour'] = train['Time'].apply(GetHour)

    train['Date'] = pd.to_datetime(train['Date'])
    train['Month'] = train['Date'].apply(GetMonth)


    #Test Times
    test['Time'] = pd.to_datetime(test['Time'])
    test['Hour'] = test['Time'].apply(GetHour)

    test['Date'] = pd.to_datetime(test['Date'])
    test['Month'] = test['Date'].apply(GetMonth)

    #BINARY CONVERSION
    urban_rural_dict = {
        1: 0,
        2: 1
    }

    police_dict = {
        'Yes': 1,
        'No': 0
    }

    train['Urban_or_Rural_Area'] = train['Urban_or_Rural_Area'].map(urban_rural_dict)
    train['Did_Police_Officer_Attend_Scene_of_Accident'] = train['Did_Police_Officer_Attend_Scene_of_Accident'].map(police_dict)

    test['Urban_or_Rural_Area'] = test['Urban_or_Rural_Area'].map(urban_rural_dict)
    test['Did_Police_Officer_Attend_Scene_of_Accident'] = test['Did_Police_Officer_Attend_Scene_of_Accident'].map(police_dict)

    train = train.drop(columns=['Date', 'Time'])
    test = test.drop(columns=['Date', 'Time'])


    train = train.set_index('postcode')
    test = test.set_index('postcode')

    for x in train.columns:
        if x not in cat_cols + num_cols + bin_cols + sqrt_cols + cbrt_cols + log_cols + target_col:
            train.drop(columns=[x], inplace=True)
            test.drop(columns=[x], inplace=True)

    train['Number_of_Vehicles'] = train['Number_of_Vehicles']
    test['Number_of_Vehicles'] = test['Number_of_Vehicles']

    #SEPARATE TARGET AND FEATURES

    X = train.drop(columns=['Number_of_Casualties'])
    y = train[['Number_of_Casualties']]

    test_X = test.drop(columns=['Number_of_Casualties'])


    #PIPELINES
    sqrt_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='mean')),
        ('sqrt', FunctionTransformer(Sqrt)),
        ('scaler', StandardScaler())
    ])

    cbrt_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='mean')),
        ('cbrt', FunctionTransformer(Cbrt)),
        ('scaler', StandardScaler())
    ])

    log_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='mean')),
        ('log', FunctionTransformer(Log)),
        ('scaler', StandardScaler())
    ])

    num_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'))
    ])

    ct = ColumnTransformer([
    ('sqrt', sqrt_pipeline, sqrt_cols),
    ('cbrt', cbrt_pipeline, cbrt_cols),
    ('log', log_pipeline, log_cols),
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
    ], remainder='passthrough'
    )

    X_trans = ct.fit_transform(X)
    test_X = ct.transform(test_X)
    y = y.to_numpy()

    return X_trans, y, test_X
