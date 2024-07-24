# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 11:11:52 2023

@author: Robbe Neyns

This script over or undersamples the dataset. Previously, the oversampling was done using a pytorch module
but to create more control over the dataset, this script applies the sampling to the .csv file containing the tree samples.
"""

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import imblearn


def resample(df_i,sampling="under",num_classes=2, NearMissV = 3, seed = None):
    df = df_i #pd.read_csv(df_i)
    df = df.dropna()
    print(df.isnull().sum().sum())
    print(df.loc[pd.isna(df).any(axis=1), :].index)

    #First distinguish between the training and test samples
    train = df[df["Train_test"] == 0]
    test = df[df["Train_test"] == 1]
    
    #Get the X and y data from the datafame
    y = train["essence_cat"].to_numpy()
    train.drop(['essence_cat'],inplace = True,axis=1)
    X = train.to_numpy()
    
    #Perform the sampling
    #for oversampling
    if sampling == 'over':
      oversample = RandomOverSampler(sampling_strategy = 'minority', random_state=seed)
      
      for i in range(num_classes):
          X,y = oversample.fit_resample(X, y)
    
    #for undersampling
    
    # define undersample strategy
    if sampling == 'under':
      undersample = RandomUnderSampler(sampling_strategy='majority', random_state=seed)
    
      for i in range(num_classes):
          X,y = undersample.fit_resample(X, y)
          
    if sampling == "NearMiss":
        
        undersample = imblearn.under_sampling.NearMiss(version=NearMissV)

        X,y = undersample.fit_resample(X, y)
        
    if sampling == "CNN":
        # define the undersampling method
        undersample = imblearn.under_sampling.CondensedNearestNeighbour(n_neighbors=1)
        X,y = undersample.fit_resample(X, y)
        
    if sampling == "Tomek":
        undersample = imblearn.under_sampling.TomekLinks()
        
        X,y = undersample.fit_resample(X, y)
        
        undersample = imblearn.under_sampling.CondensedNearestNeighbour(n_neighbors=1)
        X,y = undersample.fit_resample(X, y)
        
    if sampling == "OSS":
        # define the undersampling method
        undersample = imblearn.under_sampling.OneSidedSelection(n_neighbors=1, n_seeds_S=200)
        X,y = undersample.fit_resample(X, y)
        
    if sampling == "NCR":
        undersample = imblearn.under_sampling.NeighbourhoodCleaningRule(n_neighbors=3, threshold_cleaning=0.5)
        X,y = undersample.fit_resample(X, y)

    #Add the new training set to the dataset
    new_train = pd.DataFrame(X, columns=train.columns)
    new_train["essence_cat"] = y
    result_df = pd.concat([new_train, test], ignore_index=True)
    return result_df



