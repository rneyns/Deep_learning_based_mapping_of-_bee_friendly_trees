# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 11:06:01 2023

@author: Robbe Neyns
"""

import torch
from torch import nn
from models import SAINT, SAINT_vision

import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import count_parameters, classification_scores, mean_sq_error, class_wise_acc_
from augmentations_extra_embedding import embed_data_mask
from augmentations import add_noise
from data_prep_extra_embedding import data_prep, DataSetCatCon, data_prep_premade
from train_val_test_div_2 import train_val_test_div_2 
from Over_Undersampling import resample
import pandas as pd 
import imblearn
from read_and_merge_datasets import RandM
import ast

import os
import numpy as np
parser = argparse.ArgumentParser()

parser.add_argument('--dset_id', required=True, type=str)
parser.add_argument('--predset_id', required=False, type=str)
parser.add_argument('--output_name', required=True, type=str)
parser.add_argument('--vision_dset', action = 'store_true')
parser.add_argument('--task', required=True, default = 'clf' , type=str,choices = ['binary','multiclass','regression','clf'])
parser.add_argument('--cont_embeddings', default='spatio-temporal', type=str,choices = ['MLP','Noemb','pos_singleMLP','spatio-temporal','temporal'])
parser.add_argument('--embedding_size', default=32, type=int)
parser.add_argument('--transformer_depth', default=6, type=int)
parser.add_argument('--attention_heads', default=8, type=int)
parser.add_argument('--attention_dropout', default=0.1, type=float)
parser.add_argument('--ff_dropout', default=0.1, type=float)
parser.add_argument('--attentiontype', default='colrow', type=str,choices = ['col','colrow','row','justmlp','attn','attnmlp'])
parser.add_argument('--fixed_train_test', action='store_true')
parser.add_argument('--undersample', action='store_true')
parser.add_argument('--spatio_temp', action='store_true')
parser.add_argument('--transfer_learning', action='store_true')
parser.add_argument('--apply_version', action='store_false')

parser.add_argument('--optimizer', default='AdamW', type=str,choices = ['AdamW','Adam','SGD'])
parser.add_argument('--scheduler', default='cosine', type=str,choices = ['cosine','linear'])

parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--batchsize', default=512, type=int)
parser.add_argument('--savemodelroot', default='./bestmodels', type=str)
parser.add_argument('--run_name', default='testrun', type=str)
parser.add_argument('--set_seed', default= 1 , type=int)
parser.add_argument('--dset_seed', default= 1 , type=int)
parser.add_argument('--active_log', action = 'store_true')

parser.add_argument('--pretrain', action = 'store_true')
parser.add_argument('--pretrain_epochs', default=22, type=int)
parser.add_argument('--pt_tasks', default=['cutmix','contrastive'], type=str,nargs='*',choices = ['contrastive','contrastive_sim','denoising'])
parser.add_argument('--pt_aug', default=[], type=str,nargs='*',choices = ['mixup','cutmix'])
parser.add_argument('--pt_aug_lam', default=0.1, type=float)
parser.add_argument('--mixup_lam', default=0.3, type=float)

parser.add_argument('--train_noise_type', default=None , type=str,choices = ['missing','cutmix'])
parser.add_argument('--train_noise_level', default=0, type=float)

parser.add_argument('--ssl_samples', default= None, type=int)
parser.add_argument('--pt_projhead_style', default='diff', type=str,choices = ['diff','same','nohead'])
parser.add_argument('--nce_temp', default=0.7, type=float)

parser.add_argument('--lam0', default=0.5, type=float)
parser.add_argument('--lam1', default=10, type=float)
parser.add_argument('--lam2', default=1, type=float)
parser.add_argument('--lam3', default=10, type=float)
parser.add_argument('--final_mlp_style', default='sep', type=str,choices = ['common','sep'])


opt = parser.parse_args()
modelsave_path = os.path.join(os.getcwd(),opt.savemodelroot,opt.task,str(opt.dset_id)+"under",opt.run_name)
if opt.task == 'regression':
    opt.dtask = 'reg'
else:
    opt.dtask = 'clf'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}.")

torch.manual_seed(opt.set_seed)
os.makedirs(modelsave_path, exist_ok=True)

if opt.active_log:
    import wandb
    wandb.login(key='f746dc65fb72b570908ed1dd5b2b780d7e438243')
    if opt.train_noise_type is not None and opt.train_noise_level > 0:
        wandb.init(project="saint_v2_robustness", group =f'{opt.run_name}_{opt.task}' ,name = f'{opt.task}_{opt.train_noise_type}_{str(opt.train_noise_level)}_{str(opt.attentiontype)}_{str(opt.dset_id)}')
    elif opt.ssl_samples is not None:
        wandb.init(project="saint_v2_ssl", group = f'{opt.run_name}_{opt.task}' ,name = f'{opt.task}_{str(opt.ssl_samples)}_{str(opt.attentiontype)}_{str(opt.dset_id)}')
    else:
        raise'wrong config.check the file you are running'
    wandb.config.update(opt)

for fold in range(1,11):
    # Open the dataset files
    ############################
    
    
    #divide into training and test set 
    print(f'The fixed train test parameter is {opt.fixed_train_test}')
    dfs = []
    for df in os.listdir(opt.dset_id):
      if "Brunswick_2023" in df:
        dfs.append(pd.read_csv(opt.dset_id + "/" + df))
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
    dataset["Train_test"] = df1["Train_test"]
    
    # Add this dataframe to the final list
    print(f"dataset has shape: {dataset.shape}")
    
    #create the DOY tensor
    DOY = torch.from_numpy(np.array([[37],[45],[59],[95],[99],[108],[119],[128],[132],[149],[153],[160],[163],[187],[188],[191],[195],[248],[249],[257],[270],[297]])) #Brunswick
    #DOY = torch.from_numpy(np.array([[244],[61],[62],[248],[195],[38],[68],[69],[227],[158],[134],[106],[107],[231],[139],[167],[51],[78],[177],[263],[85],[147],[110],[150],[0]])) #Freiburg
    #DOY = torch.from_numpy(np.array([[244],[61],[62],[247],[187],[38],[68],[69],[222],[162],[134],[106],[107],[229],[138],[169],[51],[80],[175],[268],[85],[147],[118],[150],[364]])) #Munich
    #DOY = torch.from_numpy(np.array([[5],[37],[43],[59],[63],[65],[66],[67],[69],[71],[74],[81],[83],[86],[99],[100],[101],[107],[108],[113],[119],[122],[127],[128],[129],[135],[136],[137],[146],[149],[151],[153],[155],[161],[167],[171],[177],[188],[194],[197],[199],[204],[215],[230],[248],[250],[252],[257],[267],[270],[280],[290],[325]]))
    DOY_pre = DOY.repeat(len(dataset), 1, 1)
    
    #Change Train_test to binary format depending on the fold
    # Function to transform the column values
    def transform_value(x):
        try:
            if int(x) == fold:
                return 1
            else:
                return 0
        except ValueError:
            # Handle the case where conversion to int fails
            print(f"deze waarde is raar: {x}")
            return 0
        
    # Apply the function to the specific column
    dataset['Train_test'] = dataset['Train_test'].apply(transform_value)
    
    #Load the extra information
    ###################################
    satellite_azimuth = np.expand_dims(pd.read_csv(opt.dset_id + "/Braunsweigh_satellite_azimuth.csv").values, axis=-1)
    sun_azimuth = np.expand_dims(pd.read_csv(opt.dset_id + "/Braunsweigh_sun_azimuth.csv").values, axis=-1)
    sun_elevation = np.expand_dims(pd.read_csv(opt.dset_id + "/Braunsweigh_sun_elevation.csv").values, axis=-1)
    view_angle = np.expand_dims(pd.read_csv(opt.dset_id + "/Braunsweigh_view_angle.csv").values, axis=-1)
    
    
    DOY = DOY.repeat(len(dataset), 1, 1)
    
    #Print info about dataset and get number of continuous variables
    ################################################################
    
    num_rows_with_1 = (dataset['essence_cat'] == 1).sum()
    ##### Count the number of rows with value 0
    num_rows_with_0 = (dataset['essence_cat'] == 0).sum()
    
    print("Number of rows with value 1:", num_rows_with_1)
    print("Number of rows with value 0:", num_rows_with_0)

        
    num_continuous = (dataset.shape[1]-3) * 4
    
    ##### Count the number of rows with value 1
    num_rows_with_1_test = (dataset['Train_test'] == 1).sum()
    
    ##### Count the number of rows with value 0
    num_rows_with_0_test = (dataset['Train_test'] == 0).sum()
    
    print("Number of rows with value in traintest 1:", num_rows_with_1_test)
    print("Number of rows with value in traintest 0:", num_rows_with_0_test)
    
    #### Defining the weights simply as the inverse frequency of each class and rescaling to the number of classes
    ##############################################################################################################
    
    w0 = 1/(num_rows_with_0/(num_rows_with_0+num_rows_with_1))
    w1 = 1/(num_rows_with_1/(num_rows_with_0+num_rows_with_1))
    w0_norm = (w0 / (w0+w1))*2
    w1_norm = (w1 / (w0+w1))*2
      
    ##### Initializing the dataloaders
    ###################################
    
    print('---- Initializing the dataloaders ----')
    
    data_prepped = data_prep_premade(ds_id=dataset, DOY = DOY, satellite_azimuth = satellite_azimuth, sun_azimuth = sun_azimuth, sun_elevation = sun_elevation, view_angle = view_angle, seed = opt.dset_seed, task=opt.task)
    cat_dims = data_prepped[0]
    cat_idxs = data_prepped[1]
    con_idxs = data_prepped[2]
    X_train = data_prepped[3]
    y_train = data_prepped[4]
    ids_train = data_prepped[5] 
    X_valid = data_prepped[6]
    y_valid = data_prepped[7]
    ids_valid = data_prepped[8]
    X_test = data_prepped[9]
    y_test = data_prepped[10]
    ids_test = data_prepped[11]
    train_mean = data_prepped[12] 
    train_std = data_prepped[13]
    DOY_train = data_prepped[14]
    DOY_valid = data_prepped[15]
    satellite_azimuth_train = data_prepped[16]
    satellite_azimuth_valid = data_prepped[17]
    sun_azimuth_train = data_prepped[18]
    sun_azimuth_valid = data_prepped[19]
    sun_elevation_train = data_prepped[20]
    sun_elevation_valid = data_prepped[21]
    view_angle_train = data_prepped[22]
    view_angle_valid = data_prepped[23]
    
    
    
    continuous_mean_std = np.array([train_mean,train_std]).astype(np.float32) 
    print(f"X_valid shape before after the thingy {X_valid['data'].shape}")
    print(f"con_idxs: {con_idxs}")
    
    ##### Setting some hyperparams based on inputs and dataset
    _,nfeat,nbands = X_train['data'].shape
    print(f"Number of dates: {nfeat}; and number of bands: {nbands}")
    
    if nfeat > 100:
        opt.embedding_size = min(4,opt.embedding_size)
        #The batch size needs to be at least  to make optimal use of the intersample attention
        opt.batchsize = min(64, opt.batchsize)
    if opt.attentiontype != 'col':
        opt.transformer_depth = 1
        opt.attention_heads = 4
        opt.attention_dropout = 0.8
        opt.embedding_size = 16
        if opt.optimizer =='SGD':
            opt.ff_dropout = 0.4
            opt.lr = 0.01
        else:
            opt.ff_dropout = 0.8
    
    
    train_ds = DataSetCatCon(X_train, y_train, DOY_train, satellite_azimuth_train, sun_azimuth_train, sun_elevation_train, view_angle_train,  ids_train, cat_idxs, opt.dtask)#, continuous_mean_std)
    trainloader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True,num_workers=1)
    
    valid_ds = DataSetCatCon(X_valid, y_valid, DOY_valid, satellite_azimuth_valid, sun_azimuth_valid, sun_elevation_valid, view_angle_valid, ids_valid, cat_idxs, opt.dtask)#, continuous_mean_std)
    validloader = DataLoader(valid_ds, batch_size=opt.batchsize, shuffle=False,num_workers=1)
    
    test_ds = DataSetCatCon(X_valid, y_valid, DOY_valid, satellite_azimuth_valid, sun_azimuth_valid, sun_elevation_valid, view_angle_valid, ids_valid, cat_idxs, opt.dtask)#, continuous_mean_std)
    testloader = DataLoader(test_ds, batch_size=opt.batchsize, shuffle=False,num_workers=1)
    
    if opt.task == 'regression':
        y_dim = 1
    else:
        y_dim = len(np.unique(y_train['data'][:,0]))
        print(y_dim)
    
    cat_dims = np.append(np.array([1]),np.array(cat_dims)).astype(int) #Appending 1 for CLS token, this is later used to generate embeddings.
    
    
    model = SAINT(
    categories = tuple(cat_dims), 
    num_continuous = num_continuous,                
    dim = opt.embedding_size,                           
    dim_out = 1,                       
    depth = opt.transformer_depth,                       
    heads = opt.attention_heads,                         
    attn_dropout = opt.attention_dropout,             
    ff_dropout = opt.ff_dropout,                  
    mlp_hidden_mults = (4, 2),       
    cont_embeddings = opt.cont_embeddings,
    attentiontype = opt.attentiontype,
    final_mlp_style = opt.final_mlp_style,
    y_dim = y_dim,
    d_x = 5
    )
    
    vision_dset = opt.vision_dset
    
    if y_dim == 2 and opt.task == 'binary':
        # opt.task = 'binary'
        print(f'-----Defining a binary CrossEntropyLoss function with weights {w0_norm} and {w1_norm}')
        criterion = nn.CrossEntropyLoss(torch.Tensor([w0_norm,w1_norm])).to(device)
    elif y_dim > 2:
        # opt.task = 'multiclass'
        criterion = nn.CrossEntropyLoss().to(device)
    elif opt.task == 'regression':
        criterion = nn.MSELoss().to(device)
    else:
        raise'case not written yet'
    
    
    model.to(device)
    
    ## Choosing the optimizer
    if opt.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=opt.lr,
                              momentum=0.9, weight_decay=5e-4)
        from utils import get_scheduler
        scheduler = get_scheduler(opt, optimizer)
    elif opt.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(),lr=opt.lr)
    elif opt.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(),lr=opt.lr)
    
    best_valid_auroc = 0
    best_valid_accuracy = 0
    best_test_auroc = 0
    best_kappa = 0
    best_test_accuracy = 0
    best_valid_rmse = 100000
    print('----Training begins now----')
    for epoch in range(opt.epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            optimizer.zero_grad()
            # x_categ is the the categorical data, with y appended as last feature. x_cont has continuous data. cat_mask is an array of ones same shape as x_categ except for last column(corresponding to y's) set to 0s. con_mask is an array of ones same shape as x_cont. 
            ids, DOY, satellite_azimuth, sun_azimuth, sun_elevation, view_angle, x_categ, x_cont, y_gts = data[0].to(device), data[1].to(device).type(torch.float32),data[2].to(device).type(torch.float32),data[3].to(device).type(torch.float32),data[4].type(torch.LongTensor).to(device),data[5].to(device).type(torch.float32),data[6].to(device).type(torch.float32), data[7].to(device).type(torch.float32), data[8].to(device).type(torch.float32)
            if opt.train_noise_type is not None and opt.train_noise_level>0:
                noise_dict = {
                    'noise_type' : opt.train_noise_type,
                    'lambda' : opt.train_noise_level
                }
                if opt.train_noise_type == 'cutmix':
                    x_categ, x_cont = add_noise(x_categ,x_cont, noise_params = noise_dict)
            # We are converting the data to embeddings in the next step
            _ , x_categ_enc, x_cont_enc, con_mask = embed_data_mask(x_categ, x_cont, model,vision_dset, DOY=DOY, satellite_azimuth=satellite_azimuth, sun_azimuth=sun_azimuth, sun_elevation=sun_elevation, view_angle=view_angle)           
            reps = model.transformer(x_categ_enc, x_cont_enc, con_mask)
            # select only the representations corresponding to y and apply mlp on it in the next step to get the predictions.
            y_reps = reps[:,0,:]
            y_outs = model.mlpfory(y_reps)
            if opt.task == 'regression':
                loss = criterion(y_outs,y_gts) 
            else:
                loss = criterion(y_outs,y_gts.squeeze()) 
            loss.backward()
            optimizer.step()
            if opt.optimizer == 'SGD':
                scheduler.step()
            running_loss += loss.item()
        # print(running_loss)
        if opt.active_log:
            wandb.log({'epoch': epoch ,'train_epoch_loss': running_loss, 
            'loss': loss.item()
            })
        if epoch%5==0:
                model.eval()
                with torch.no_grad():
                    if opt.task in ['binary','multiclass']:
                        accuracy, auroc, kappa = classification_scores(model, validloader, device, opt.task,vision_dset)
                        test_accuracy, test_auroc, test_kappa = classification_scores(model, testloader, device, opt.task,vision_dset)
                        acc_classwise, conf_matrix = class_wise_acc_(model,validloader,device)
                        print('[EPOCH %d] VALID ACCURACY: %.3f, VALID AUROC: %.3f, VALID KAPPA: %.3f' %
                            (epoch + 1, accuracy,auroc,kappa ))
                        print('[EPOCH %d] TEST ACCURACY: %.3f, TEST AUROC: %.3f, TEST KAPPA: %.3f' %
                            (epoch + 1, test_accuracy,test_auroc, test_kappa ))
                        print(f"class_wise_accuracies: {acc_classwise}")
                        print(f"confusion matrix: {conf_matrix}")
                        if opt.active_log:
                            wandb.log({'valid_accuracy': accuracy ,'valid_auroc': auroc })     
                            wandb.log({'test_accuracy': test_accuracy ,'test_auroc': test_auroc })  
                        if opt.task =='multiclass':
                            if accuracy > best_valid_accuracy:
                                best_valid_accuracy = accuracy
                                best_test_auroc = test_auroc
                                best_test_accuracy = test_accuracy
                        else:
                            if test_kappa > best_kappa:
                                best_kappa = test_kappa
                                best_test_auroc = test_auroc
                                best_test_accuracy = test_accuracy               
    
                    else:
                        valid_rmse = mean_sq_error(model, validloader, device, vision_dset)    
                        test_rmse = mean_sq_error(model, testloader, device, vision_dset)  
                        print('[EPOCH %d] VALID RMSE: %.3f' %
                            (epoch + 1, valid_rmse ))
                        print('[EPOCH %d] TEST RMSE: %.3f' %
                            (epoch + 1, test_rmse ))
                        if opt.active_log:
                            wandb.log({'valid_rmse': valid_rmse ,'test_rmse': test_rmse })     
                        if valid_rmse < best_valid_rmse:
                            best_valid_rmse = valid_rmse
                            best_test_rmse = test_rmse
                model.train()
                    
    
    total_parameters = count_parameters(model)
    print('TOTAL NUMBER OF PARAMS: %d' %(total_parameters))
    if opt.task =='binary':
        print('Kappa on best model:  %.3f' %(best_kappa))
    elif opt.task =='multiclass':
        print('Accuracy on best model:  %.3f' %(best_test_accuracy))
    else:
        print('RMSE on best model:  %.3f' %(best_test_rmse))
    
    if opt.active_log:
        if opt.task == 'regression':
            wandb.log({'total_parameters': total_parameters, 'test_rmse_bestep':best_test_rmse , 
            'cat_dims':len(cat_idxs) , 'con_dims':len(con_idxs) })        
        else:
            wandb.log({'total_parameters': total_parameters, 'test_auroc_bestep':best_test_auroc , 
            'test_accuracy_bestep':best_test_accuracy,'cat_dims':len(cat_idxs) , 'con_dims':len(con_idxs) })