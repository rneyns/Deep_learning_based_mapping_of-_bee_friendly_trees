# -*- coding: utf-8 -*-
"""
Created on Tue May 14 13:26:05 2024

@author: Robbe Neyns
"""
import os
import pandas as pd
import numpy as np
import torch
from train_val_test_div_2 import train_val_test_div_2

def RandM(folder,shifted=True):
    # Define the DOY for every city, this could be included as an argument in the future
    # Brunswick
    DOY_1 = torch.from_numpy(np.array([[37],[45],[59],[71],[76],[78],[80],[82],[95],[99],[107],[108],[119],[123],[124],[128],[132],[153],[160],[163],[165],[168],[187],[188],[195],[208],[223],[235],[243],[246],[248],[249],[257],[270],[276],[281],[284],[297],[302],[322]]))
    #Munich
    DOY_2 = torch.from_numpy(np.array([[0],[38],[39],[51],[61],[62],[68],[69],[78],[83],[84],[85],[100],[106],[107],[109],[110],[128],[130],[134],[139],[141],[147],[150],[152],[158],[165],[167],[176],[177],[195],[198],[227],[231],[234],[236],[244],[248],[252],[255],[263],[264],[265],[297],[299],[310],[318],[360],[363]]))
    #Freiburg
    DOY_3 = torch.from_numpy(np.array([[38],[51],[61],[62],[64],[68],[69],[72],[80],[85],[86],[94],[106],[107],[118],[129],[134],[138],[147],[150],[162],[169],[175],[187],[221],[222],[229],[244],[247],[268],[364]]))
    #Berlin
    DOY_4 = torch.from_numpy(np.array([[5],[37],[43],[59],[63],[65],[66],[67],[69],[71],[74],[81],[83],[86],[99],[100],[101],[107],[108],[113],[119],[122],[127],[128],[129],[135],[136],[137],[146],[149],[151],[153],[155],[161],[167],[171],[177],[188],[194],[197],[199],[204],[215],[230],[248],[250],[252],[257],[267],[270],[280],[290],[325]]))
    #Getting the number of dates in the Berlin tensor because it is the longest one
    #required_dim2_size = DOY_4.size(0)
    if shifted:
      #Shift based on the SOS of Salix 
      DOY_1 -= 86
      DOY_2 -= 90
      DOY_3 -= 74
      DOY_4 -= 81 
    required_dim2_size = max([DOY_1.size(0),DOY_2.size(0),DOY_3.size(0),DOY_4.size(0)])
    print(f"required dim size: {required_dim2_size}")
    
    dfs_f = []
    Brunswick = []
    Munich = []
    Freiburg = []
    Berlin = []
    cities = [Brunswick,Munich,Freiburg,Berlin]
  

    for file in os.listdir(folder):
        df = pd.read_csv(folder + "/" + file)
        #prepare the DOY's 
        if 'Brunswick' in file:
            cities[0].append(df)
            DOY_1n = DOY_1.repeat(len(df), 1, 1)
        elif 'Munich' in file:
            cities[1].append(df)
            DOY_2n = DOY_2.repeat(len(df), 1, 1)
        elif 'Freiburg' in file:
            cities[2].append(df)
            DOY_3n = DOY_3.repeat(len(df), 1, 1)
        elif 'Berlin' in file:
            cities[3].append(df)
            DOY_4n = DOY_4.repeat(len(df), 1, 1)
            
    #Pad the DOYS
    DOYS = [DOY_1n,DOY_2n,DOY_3n,DOY_4n]

    DOYS_padded = []
    for DOY in DOYS:
        # Create a new tensor with the desired shape filled with NaN values
        padded_tensor = torch.full((DOY.size(0), required_dim2_size,DOY.size(2)), float('nan'))
        print(f"size of the padded tensor: {padded_tensor.shape}")
        # Copy the original tensor values into the new tensor
        padded_tensor[:, :DOY.size(1),:] = DOY
        DOYS_padded.append(padded_tensor)
      
    combined_DOY = torch.cat(DOYS_padded, dim=0)
    
    for dfs in cities:
        # Initialize a new DataFrame with the same structure
        df1 = dfs[0]
        dataset = pd.DataFrame(index=df1.index, columns=df1.columns)
        
        # Populate the new DataFrame with lists from corresponding cells of all DataFrames
        for col in dataset.columns:
            for idx in dataset.index: 
                dataset.at[idx, col] = [df.at[idx, col] for df in dfs]  # Create a list of values from all dataframes
                
        #make sure that the id and label column is not a list
        dataset["essence_cat"] = df1["essence_cat"]
        dataset["id"] = df1["id"]
        dataset = train_val_test_div_2(dataset,"essence_cat")
        
        # Add this dataframe to the final list
        print(f"dataset has shape: {dataset.shape}")
        dfs_f.append(dataset)
    
    counter = 0
    for df in dfs_f: 
        # Create a dictionary that maps current column names to new names
        new_column_names = {old_name: new_name for new_name, old_name in enumerate(df.columns)}
        new_column_names.pop("essence_cat")
        new_column_names.pop("id")
        new_column_names.pop("Train_test")

        # Rename the columns
        df.rename(columns=new_column_names, inplace=True)

        #if counter == 2: 
        #  df['Train_test'] = 1
        #else:
        #  df['Train_test'] = 0
        #counter += 1

    combined_df = pd.concat(dfs_f, axis=0, ignore_index=True, sort=False)

    return combined_df, combined_DOY

            

    #merge the different dataframes
    
        