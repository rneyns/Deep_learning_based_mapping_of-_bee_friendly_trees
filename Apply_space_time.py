# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 09:23:31 2023

@author: Robbe Neyns
"""

import torch
from models import SAINT
import argparse
from torch.utils.data import DataLoader
from augmentations import embed_data_mask
from data_prep import  DataSetCatCon, data_prep_premade
import pandas as pd 
import os
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--dset_id', required=True, type=str)
parser.add_argument('--output_name', required=True, type=str)
parser.add_argument('--labelHeader', required=True, type=str)
parser.add_argument('--IDHeader', required=True, type=str)
parser.add_argument('--vision_dset', action = 'store_true')
parser.add_argument('--model_weights', required=True, type=str)
parser.add_argument('--task', required=True, default = 'binary' , type=str,choices = ['binary','multiclass','regression','clf'])
parser.add_argument('--cont_embeddings', default='MLP', type=str,choices = ['MLP','Noemb','pos_singleMLP'])
parser.add_argument('--embedding_size', default=32, type=int)
parser.add_argument('--transformer_depth', default=6, type=int)
parser.add_argument('--attention_heads', default=8, type=int)
parser.add_argument('--attention_dropout', default=0.1, type=float)
parser.add_argument('--ff_dropout', default=0.1, type=float)
parser.add_argument('--attentiontype', default='colrow', type=str,choices = ['col','colrow','row','justmlp','attn','attnmlp'])
parser.add_argument('--torchscript', default=False)

parser.add_argument('--batchsize', default=50, type=int)

parser.add_argument('--pt_tasks', default=['contrastive','denoising'], type=str,nargs='*',choices = ['contrastive','contrastive_sim','denoising'])
parser.add_argument('--dset_seed', default= 1 , type=int)
parser.add_argument('--pt_aug', default=[], type=str,nargs='*',choices = ['mixup','cutmix'])
parser.add_argument('--pt_aug_lam', default=0.1, type=float)
parser.add_argument('--mixup_lam', default=0.3, type=float)

parser.add_argument('--ssl_samples', default= None, type=int)
parser.add_argument('--pt_projhead_style', default='diff', type=str,choices = ['diff','same','nohead'])
parser.add_argument('--nce_temp', default=0.7, type=float)

parser.add_argument('--lam0', default=0.5, type=float)
parser.add_argument('--lam1', default=10, type=float)
parser.add_argument('--lam2', default=1, type=float)
parser.add_argument('--lam3', default=10, type=float)
parser.add_argument('--final_mlp_style', default='sep', type=str,choices = ['common','sep'])


opt = parser.parse_args()

if opt.task == 'regression':
    opt.dtask = 'reg'
else:
    opt.dtask = 'clf'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}.")



vision_dset = opt.vision_dset

def make_predictions(model, dataloader, device):
    model.eval()
    all_predictions = []
    idxs = []
    correct = []
    ys = []
    with torch.no_grad():
        for data in dataloader:
            ids, DOY, x_categ, x_cont, y_gts = data[0].to(device), data[1].to(device).type(torch.float32),data[2].to(device).type(torch.float32),data[3].to(device).type(torch.float32),data[4].type(torch.LongTensor).to(device)#,data[5].to(device).type(torch.float32)#,data[6].to(device).type(torch.float32)
            _ , x_categ_enc, x_cont_enc, con_mask = embed_data_mask(x_categ, x_cont, model,vision_dset,DOY=DOY)           
            reps = model.transformer(x_categ_enc, x_cont_enc, con_mask)
            y_reps = reps[:, 0, :]
            y_outs = model.mlpfory(y_reps)
            y_label = torch.argmax(y_outs, dim=1)
            all_predictions.extend(y_label.cpu().numpy())
            idxs.extend(ids.cpu().numpy())
            ys.extend(y_gts.cpu().numpy())
            for i in range(len(y_label.cpu().numpy())):
                if y_label.cpu().numpy()[i] == y_gts.cpu().numpy()[i]:
                    correct.append(1)
                else:
                    correct.append(0)
    return idxs, np.array(all_predictions), correct, np.array(ys)

#########################################################
#--------------  Prepare the dataset--------------------#
#########################################################


dfs = []
for df in os.listdir(opt.dset_id):
  if "Brunswick" in df:
    dfs.append(pd.read_csv(opt.dset_id + "/" + df))
    
# Initialize a new DataFrame with the same structure
df1 = dfs[0]
dataset = pd.DataFrame(index=df1.index, columns=df1.columns)

# Populate the new DataFrame with lists from corresponding cells of all DataFrames
for col in dataset.columns:
    for idx in dataset.index: 
        dataset.at[idx, col] = [df.at[idx, col] for df in dfs]  # Create a list of values from all dataframes
        
#make sure that the id and label column is not a list
dataset["id"] = df1["id"]

# Add this dataframe to the final list
print(f"dataset has shape: {dataset.shape}")

#create the DOY tensor
DOY = torch.from_numpy(np.array([[37],[45],[59],[71],[76],[78],[80],[82],[95],[99],[107],[108],[119],[123],[124],[128],[132],[153],[160],[163],[165],[168],[187],[188],[195],[208],[223],[235],[243],[246],[248],[249],[257],[270],[276],[281],[284],[297],[302],[322]])) #Brunswick
DOY = DOY.repeat(len(dataset), 1, 1)

print(f"DOY shape: {DOY.shape}")

dataset["Train_test"] = 0
dataset["essence_cat"] = 0
dataset.to_csv("merged_dataset.csv")
dataset = dataset.iloc[:10000]

print(f"dataset shape: {dataset.shape}")
num_continuous = (dataset.shape[1]-3) * 4
print(f"number of continous variables: {num_continuous}")
dataset = dataset.reset_index(drop=True)
print(133)
cat_dims_pre, cat_idxs_pre, con_idxs_pre, X_train_pre, y_train_pre, ids_train_pre, X_valid_pre, y_valid_pre, ids_valid_pre, X_test_pre, y_test_pre, ids_test_pre, train_mean_pre, train_std_pre, DOY_train_pre, DOY_valid_pre = data_prep_premade(ds_id=dataset, DOY = DOY, seed = opt.dset_seed, task=opt.task)
print(X_train_pre['data'].shape)
#continuous_mean_std = np.array([train_mean,train_std]).astype(np.float32) 
ds = DataSetCatCon(X_train_pre, y_train_pre, DOY_train_pre, ids_train_pre, cat_idxs_pre, opt.dtask)#, continuous_mean_std=continuous_mean_std)

predictloader = DataLoader(ds, batch_size=opt.batchsize, shuffle=False,num_workers=1)

##### Setting some hyperparams based on inputs and dataset
nfeat = ds.get_size_X()
if nfeat > 100:
    opt.embedding_size = min(4,opt.embedding_size)
    opt.batchsize = min(64, opt.batchsize)
if opt.attentiontype != 'col':
    opt.transformer_depth = 1
    opt.attention_heads = 4
    opt.attention_dropout = 0.8
    opt.embedding_size = 16
    opt.ff_dropout = 0.8

print(153)
opt.batchsize = 128

y_dim = 2

cat_dims = np.append(np.array([1]),np.array([])).astype(int) #Appending 1 for CLS token, this is later used to generate embeddings.

if opt.torchscript:
    model = torch.jit.load('model_scripted.pt')
    model.eval()
else:
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
    y_dim = y_dim
    )
    
    
    
    # print(count_parameters(model))
    # import ipdb; ipdb.set_trace()
    print("CHECK if the parameters in the build and loaded model are compatible...")
    # 3. Specify the file path of the saved model checkpoint
    checkpoint_path = opt.model_weights
    
    # 4. Load the model checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)  # Use 'cuda' if your model was trained on GPU
    
    # 5. Check if the keys match
    model_state_dict_keys = set(model.state_dict().keys())
    checkpoint_keys = set(checkpoint.keys())
    
    # Compare the keys
    missing_keys = model_state_dict_keys - checkpoint_keys
    unexpected_keys = checkpoint_keys - model_state_dict_keys
    
    # Print the result
    if not missing_keys and not unexpected_keys:
        print("Model weights loaded successfully.")
    else:
        if missing_keys:
            print("Missing keys in loaded checkpoint:", missing_keys)
        if unexpected_keys:
            print("Unexpected keys in loaded checkpoint:", unexpected_keys)
    
    print('APPLYING MODEL...')
    model
    model.load_state_dict(torch.load(opt.model_weights, map_location=device)) #torch load state dict automatically loads the model to CPU, so i have to first load the weights and then put on device or I have to define the "map_location" parameter
    print('____SAVED MODEL WEIGHTS____')
    print(torch.load(opt.model_weights))
    model.to(device) 

# Make predictions
idxs, ys, predictions, correct = make_predictions(model, predictloader, device)


# Create a DataFrame with the predictions
d = {'idx':idxs,'Prediction':predictions,'ys':ys,'correct':correct}
df = pd.DataFrame(data=d)

# Save the predictions to a CSV file
df.to_csv("/content/drive/MyDrive/Bee mapping spacetimeformer/output_files/" + opt.output_name, index=False)

