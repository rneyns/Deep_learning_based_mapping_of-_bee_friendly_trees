# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 11:54:36 2023

@author: Robbe Neyns
"""


import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


def simple_lapsed_time(text, lapsed):
    hours, rem = divmod(lapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(text+": {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


def task_dset_ids(task):
    dataset_ids = {
        'binary': [1487,44,1590,42178,1111,31,42733,1494,1017,4134],
        'multiclass': [188, 1596, 4541, 40664, 40685, 40687, 40975, 41166, 41169, 42734],
        'regression':[541, 42726, 42727, 422, 42571, 42705, 42728, 42563, 42724, 42729]
    }

    return dataset_ids[task]

def concat_data(X,y):
    # import ipdb; ipdb.set_trace()
    return pd.concat([pd.DataFrame(X['data']), pd.DataFrame(y['data'][:,0].tolist(),columns=['target'])], axis=1)


def data_split(X,y,ids, DOY, satellite_azimuth, sun_azimuth, sun_elevation, view_angle, indices):
    x_d = {
        'data': X.values[indices],
        'mask': X.values[indices] #np.isnan(X.values[indices]).astype(int)
    }
    
    if x_d['data'].shape != x_d['mask'].shape:
        raise'Shape of data not same as that of nan mask!'
        
    y_d = {
        'data': y[indices].reshape(-1, 1)
    } 
    id_d = {
        'id': ids[indices]
        }
    doy_d = {
        'data': DOY[indices]
        }
    satellite_azimuth_d = {
        'data': satellite_azimuth.values[indices]
        }
    sun_azimuth_d = {
        'data': sun_azimuth.values[indices]
        } 
    sun_elevation_d = {
        'data': sun_elevation.values[indices]
        }
    view_angle_d = {
        'data': view_angle.values[indices]
        }
    return x_d, y_d, id_d, doy_d, satellite_azimuth_d, sun_azimuth_d, sun_elevation_d, view_angle_d


def data_prep(ds_id, seed, task, datasplit=[.65, .15, .2], pretraining=False):
    
    np.random.seed(seed) #this is a utility function supporting older numpy code to provide a seed for a random number generator --> this makes sure that the random numbers that are generated with randn will always be the same 
    dataset = pd.read_csv(ds_id)
    
    if pretraining: 
        y = dataset["essence_cat"]
        ids = dataset["id"]
        X = dataset.drop(['id','essence_cat'], axis=1)
    else:
        y = dataset["essence_cat"]
        ids = dataset["id"]
        X = dataset.drop(['id','essence_cat','Train_test'], axis=1)

    attribute_names = X.columns.to_list()
    categorical_indicator = [False for x in range(len(attribute_names))]
    
    categorical_columns = X.columns[list(np.where(np.array(categorical_indicator)==True)[0])].tolist()
    cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))

    cat_idxs = list(np.where(np.array(categorical_indicator)==True)[0])
    con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))

    for col in categorical_columns:
        X[col] = X[col].astype("object")

    X["Set"] = np.random.choice(["train", "valid", "test"], p = datasplit, size=(X.shape[0],))

    train_indices = X[X.Set=="train"].index
    valid_indices = X[X.Set=="valid"].index
    test_indices = X[X.Set=="test"].index

    X = X.drop(columns=['Set'])
    temp = X.fillna("MissingValue")
    nan_mask = temp.ne("MissingValue").astype(int)
    
    cat_dims = []
    for col in categorical_columns:
    #     X[col] = X[col].cat.add_categories("MissingValue")
        X[col] = X[col].fillna("MissingValue")
        l_enc = LabelEncoder() 
        X[col] = l_enc.fit_transform(X[col].values)
        cat_dims.append(len(l_enc.classes_))
    for col in cont_columns:
    #     X[col].fillna("MissingValue",inplace=True)
        X.fillna(X.loc[train_indices, col].mean(), inplace=True) #We just fill up with the mean in the continuous columns whereever we have NaN
    y = y.values
    if task != 'regression':
        l_enc = LabelEncoder() 
        y = l_enc.fit_transform(y)
    X_train, y_train, ids_train, DOY_train = data_split(X,y, ids, nan_mask,train_indices)
    X_valid, y_valid, ids_valid, DOY_valid = data_split(X,y, ids, nan_mask,valid_indices)
    X_test, y_test, ids_test, DOY_test = data_split(X,y, ids, nan_mask,test_indices)

    #Make sure the last dimension (the bands) are in the right format
    X_train = np.array([np.array([np.array(xi) for xi in x]) for x in X_train['data']])
    X_valid = np.array([np.array([np.array(xi) for xi in x]) for x in X_valid['data']])
    X_test = np.array([np.array([np.array(xi) for xi in x]) for x in X_test['data']])


    train_mean, train_std = np.array(X_train['data'][:,con_idxs],dtype=np.float32).mean(0), np.array(X_train['data'][:,con_idxs],dtype=np.float32).std(0) #I think the "data" has to go here because there is no such overlapping column name in my dataset --> dus toch niet, de data werkt
    #train_mean, train_std = np.array(X_train[:,con_idxs],dtype=np.float32).mean(0), np.array(X_train[:,con_idxs],dtype=np.float32).std(0) 
    train_std = np.where(train_std < 1e-6, 1e-6, train_std)
    # import ipdb; ipdb.set_trace()
    return cat_dims, cat_idxs, con_idxs, X_train, y_train, ids_train, X_valid, y_valid, ids_valid, X_test, y_test, ids_test, train_mean, train_std, DOY_train, DOY_valid

def fill_missing(arr, placeholder="MissingValue"):
# Replace None with 'MissingValue' in array
    return np.array([x if x is not None else placeholder for x in arr])

def create_mask(arr, original_placeholder="MissingValue"):
# Create a mask where 1 is non-missing and 0 is missing
    return np.array([1 if x != original_placeholder else 0 for x in arr])


# Define the replacement function
def replace_nan_with_list(x):
    if not isinstance(x, list):
      if pd.isna(x):
          return [np.nan, np.nan, np.nan, np.nan]
    return x

def print_non_4_lists(x):
    print(len(x))
    if len(x)>= 4:
      print(x)
    return x


def data_prep_premade(ds_id, DOY, satellite_azimuth, sun_azimuth, sun_elevation, view_angle, seed, task, pretraining=False):
    
    np.random.seed(seed) #this is a utility function supporting older numpy code to provide a seed for a random number generator --> this makes sure that the random numbers that are generated with randn will always be the same 
    #dataset = pd.read_csv(ds_id)
    dataset = ds_id
    print(dataset.columns)
    
    if not pretraining:
        y = dataset["essence_cat"]
        ids = dataset['id'].values
        X = dataset.drop(['id','essence_cat','Train_test'], axis=1)
    else:
        y = dataset.iloc[:,0] #setting just the first column which actually sets reflectance values but that's ok because they won't be used during pretrainig
        ids = dataset['id'].values
        X = dataset.drop(['id','essence_cat','Train_test'], axis=1)
    

    print(f"ids[0]: {ids[0]}")
    attribute_names = X.columns.to_list()
    categorical_indicator = [False for x in range(len(attribute_names))]
    
    categorical_columns = X.columns[list(np.where(np.array(categorical_indicator)==True)[0])].tolist()
    cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))

    cat_idxs = list(np.where(np.array(categorical_indicator)==True)[0])
    con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))

    for col in categorical_columns:
        X[col] = X[col].astype("object")

    #X["Set"] = np.random.choice(["train", "valid", "test"], p = datasplit, size=(X.shape[0],))

    train_indices = dataset[dataset.Train_test==0].index
    valid_indices = dataset[dataset.Train_test==1].index
    test_indices = dataset[dataset.Train_test==1].index

    #X = X.drop(columns=['Set'])
    X =  X.applymap(replace_nan_with_list)#X.applymap(lambda x: fill_missing(x)) 
    #X = X.applymap(print_non_4_lists)


    cat_dims = []
    for col in categorical_columns:
    #     X[col] = X[col].cat.add_categories("MissingValue")
        X[col] = X[col].fillna("MissingValue")
        l_enc = LabelEncoder() 
        X[col] = l_enc.fit_transform(X[col].values)
        cat_dims.append(len(l_enc.classes_))
    for col in cont_columns:
      pass
    #     X[col].fillna("MissingValue",inplace=True)
       # X.fillna(X.loc[train_indices, col].mean(), inplace=True) #We just fill up with the mean in the continuous columns whereever we have NaN
    y = y.values
    if task != 'regression':
        l_enc = LabelEncoder() 
        y = l_enc.fit_transform(y)
    X_train, y_train, ids_train, DOY_train, satellite_azimuth_train, sun_azimuth_train, sun_elevation_train, view_angle_train = data_split(X,y, ids, DOY, satellite_azimuth, sun_azimuth, sun_elevation, view_angle, train_indices)
    X_valid, y_valid, ids_valid, DOY_valid, satellite_azimuth_valid, sun_azimuth_valid, sun_elevation_valid, view_angle_valid = data_split(X,y, ids, DOY, satellite_azimuth, sun_azimuth, sun_elevation, view_angle, valid_indices)
    X_test, y_test, ids_test, DOY_test, satellite_azimuth_test, sun_azimuth_test, sun_elevation_test, view_angle_test = data_split(X,y, ids, DOY, satellite_azimuth, sun_azimuth, sun_elevation, view_angle, test_indices)
  
    #Make sure the last dimension (the bands) are in the right format
    print(f"the shape of the Xtrain is: {X_train['data'].shape}")

    X_train['data'] = np.array([np.array([np.array(xi) for xi in x]) for x in X_train['data']])
    X_valid['data'] = np.array([np.array([np.array(xi) for xi in x]) for x in X_valid['data']])
    X_test['data'] = np.array([np.array([np.array(xi) for xi in x]) for x in X_test['data']])
    
    train_mean, train_std = np.array(X_train['data'],dtype=np.float32).mean(0), np.array(X_train['data'][:,con_idxs],dtype=np.float32).std(0) #I think the "data" has to go here because there is no such overlapping column name in my dataset --> dus toch niet, de data werkt
    #train_mean, train_std = np.array(X_train[:,con_idxs],dtype=np.float32).mean(0), np.array(X_train[:,con_idxs],dtype=np.float32).std(0) 
    train_std = np.where(train_std < 1e-6, 1e-6, train_std)
    # import ipdb; ipdb.set_trace()
    
    return cat_dims, cat_idxs, con_idxs, X_train, y_train, ids_train, X_valid, y_valid, ids_valid, X_test, y_test, ids_test, train_mean, train_std, DOY_train, DOY_valid, satellite_azimuth_train, satellite_azimuth_valid, sun_azimuth_train, sun_azimuth_valid, sun_elevation_train, sun_elevation_valid, view_angle_train, view_angle_valid


class DataSetCatCon(Dataset):
    def __init__(self, X, Y, DOY, satellite_azimuth, sun_azimuth, sun_elevation, view_angle, ids, cat_cols,task='clf',continuous_mean_std=None):
        
        cat_cols = list(cat_cols)
        X_mask =  X['mask'].copy()
        X = X['data'].copy()
        #self.ids = ids['id']
        self.n_rows = 241
        self.DOY = DOY['data']
        self.satellite_azimuth = satellite_azimuth['data']
        self.sun_azimuth = sun_azimuth['data']
        self.sun_elevation = sun_elevation['data']
        self.view_angle = view_angle['data']
        self.image_dir = "/theia/scratch/brussel/104/vsc10421/MAE_experiments"
        print(f"X shape before con_cols: {X.shape}")
        con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols))
        self.X1 = X[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2 = X[:,con_cols].copy().astype(np.float32) #numerical columns
        self.X1_mask = X_mask[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2_mask = X_mask # [:,con_cols].copy().astype(np.int64) #numerical columns
        self.ids = np.array(ids['id'])
        if task == 'clf':
            self.y = Y['data'].astype(np.intc)
        else:
            self.y = Y['data'].astype(np.intc)
        num_s, dates, bands = X.shape
        self.cls = np.zeros((num_s, 1, bands),dtype=int)
        self.cls_mask = np.ones((num_s, 1, bands),dtype=int)
        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std
        

            
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.ids[idx], self.DOY[idx], self.satellite_azimuth[idx], self.sun_azimuth[idx], self.sun_elevation[idx], self.view_angle[idx], np.concatenate((self.cls[idx], self.X1[idx])), self.X2[idx], int(self.y[idx])    
    def get_size_X(self):
        return len(self.X2[1])
    
class DataSetCatCon_apply(Dataset):
    def __init__(self, df,label_hd, id_hd, continuous_mean_std=None):
        

        self.y = df[label_hd].astype(np.intc)
            
        self.ids = df[id_hd]
        df = df.drop([label_hd,id_hd], axis=1)
        self.n_rows = 241
        self.X1 = [] #categorical columns
        self.X2 = df.values[:,:].copy().astype(np.float32) #numerical columns
        self.X2_mask = self.X2 #categorical columns
        self.X1_mask = self.X2 #categorical columns
        self.X1_mask[self.X1_mask > 0] = False
        self.X2_mask[self.X1_mask > 0] = True

        self.cls = np.zeros_like(self.y,dtype=int)
        self.cls_mask = np.ones_like(self.y,dtype=int)
        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std
        

            
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        #load the images
        # X1 has categorical data, X2 has continuous
        return np.concatenate((self.cls[idx], self.X1)), self.X2[idx], int(self.y[idx]), np.concatenate((self.cls_mask[idx], self.X1_mask[idx])), self.X2_mask[idx]
    
    def get_size_X(self):
        return len(self.X2[1])