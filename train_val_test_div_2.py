# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 10:42:47 2023

@author: Robbe Neyns

This script creates an extra column in the dataframe which indicates which samples belong to the training set (value 0) 
and which samples belong to the test set (value 1).
"""

import numpy as np
import pandas as pd

def train_val_test_div_2(file,label_column,seed=None):
    
    if not isinstance(file, pd.DataFrame):
        df = pd.read_csv(file)
    else:
        df = file
    #print(f"The dataframe contains {len(df.loc[pd.isna(df).any(axis=1), :].index)} nan rows")
    #df = df.dropna()
    y = df[label_column]
    # sampling number for each class --> vervangen door 10 % van de count van de klasse met de minste samples
    unique, counts = np.unique(y, return_counts=True)
    n1 = int(np.min(counts) * 0.1)
    
    #Make sure that there are not less than 50 samples in the validation set 
    if n1 < 50:
        n1 = 50
    
    test_idxs = []
    
    # 1. get indexes and lengths for the classes respectively
    for index, label in enumerate(unique):
        idx1 = df.index.values[df[label_column] == label]
        len1 = len(idx1)  # 1000
    
        # 2. draw index for test dataset
        draw1 = np.random.permutation(len1)[:n1]  # keep the first n1 entries to be selected
        idx_test = idx1[draw1]
        # combine the drawn indexes
        test_idxs.append(idx_test)
    idx_test = np.hstack(test_idxs)
    
    # 3. derive index for train dataset
    idx_train = df.index.difference(idx_test)
    
    # Assign the new values based on the indices
    df.loc[idx_train, "Train_test"] = 0  
    df.loc[idx_test, "Train_test"] = 1   
    
    # verify that no row was missing
    assert not df["Train_test"].isna().any()
    
    return df
