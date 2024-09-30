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
from augmentations import embed_data_mask
from augmentations import add_noise
from data_prep import data_prep, DataSetCatCon, data_prep_premade
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
parser.add_argument('--cont_embeddings', default='MLP', type=str,choices = ['MLP','Noemb','pos_singleMLP','spatio-temporal','temporal'])
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

def make_predictions(model, dataloader, device):
    softmax = nn.Softmax(dim=1)
    model.eval()
    all_predictions = []
    idxs = []
    correct = []
    ys = []
    c0_prob = []
    c1_prob = []
    counter = 0
    with torch.no_grad():
        for data in dataloader:
            ids, DOY, x_categ, x_cont, y_gts = data[0].to(device), data[1].to(device).type(torch.float32),data[2].to(device).type(torch.float32),data[3].to(device).type(torch.float32),data[4].type(torch.LongTensor).to(device)#,data[5].to(device).type(torch.float32)#,data[6].to(device).type(torch.float32)
            _ , x_categ_enc, x_cont_enc, con_mask = embed_data_mask(x_categ, x_cont, model,vision_dset,DOY=DOY)           
            reps = model.transformer(x_categ_enc, x_cont_enc, con_mask)
            y_reps = reps[:, 0, :]
            y_outs = model.mlpfory(y_reps)
            y_outs_soft = softmax(y_outs)
            y_label = torch.argmax(y_outs, dim=1)
            all_predictions.extend(y_label.cpu().numpy())
            idxs.extend(ids.cpu().numpy())
            ys.extend(y_gts.cpu().numpy())
            if counter == 1:
              print(f"y outs printed for counter 1: {y_outs}, and in cpu format for class 1: {y_outs_soft[:,1].cpu().numpy()}")
            c0_prob.extend(y_outs_soft[:,0].cpu().numpy())
            c1_prob.extend(y_outs_soft[:,1].cpu().numpy())
            for i in range(len(y_label.cpu().numpy())):
                if y_label.cpu().numpy()[i] == y_gts.cpu().numpy()[i]:
                    correct.append(1)
                else:
                    correct.append(0)
            counter +=1
    return idxs, np.array(all_predictions), correct, np.array(ys), c0_prob, c1_prob

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


# Open the dataset files
############################


#divide into training and test set 
print(f'The fixed train test parameter is {opt.fixed_train_test}')
    
# List the files in the directory
folder_path = opt.dset_id
files = os.listdir(folder_path)

# Filter files that match your criteria (assuming you only want to load files that contain 'Brunswick_cleaned_polys')
files = [file for file in files if "Brunswick_cleaned_polys" in file]

# Sort the files based on the fifth character in the filename
files_sorted = sorted(files, key=lambda x: int(x[4]))  # Sorting by the fifth character (index 4) converted to an integer
print(files_sorted)

# Initialize a list to store DataFrames
dfs = []

# Load the sorted files into the list of DataFrames
for file in files_sorted:
    df = pd.read_csv(os.path.join(folder_path, file))
    df = df.dropna()  # Drop rows with NaN values if needed
    dfs.append(df)
    
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
dataset.dropna()

# Add this dataframe to the final list
print(f"dataset has shape: {dataset.shape}")

#create the DOY tensor
DOY = torch.from_numpy(np.array([[37],[45],[59],[71],[76],[78],[80],[82],[95],[99],[107],[108],[119],[123],[124],[128],[132],[153],[160],[163],[165],[168],[187],[188],[195],[208],[223],[235],[243],[246],[248],[249],[257],[270],[276],[281],[284],[297],[302],[322]])) #Brunswick
#DOY = torch.from_numpy(np.array([[244],[61],[62],[248],[195],[38],[68],[69],[227],[158],[134],[106],[107],[231],[139],[167],[51],[78],[177],[263],[85],[147],[110],[150],[0]])) #Freiburg
#DOY = torch.from_numpy(np.array([[244],[61],[62],[247],[187],[38],[68],[69],[222],[162],[134],[106],[107],[229],[138],[169],[51],[80],[175],[268],[85],[147],[118],[150],[364]])) #Munich
#DOY = torch.from_numpy(np.array([[5],[37],[43],[59],[63],[65],[66],[67],[69],[71],[74],[81],[83],[86],[99],[100],[101],[107],[108],[113],[119],[122],[127],[128],[129],[135],[136],[137],[146],[149],[151],[153],[155],[161],[167],[171],[177],[188],[194],[197],[199],[204],[215],[230],[248],[250],[252],[257],[267],[270],[280],[290],[325]]))


#Divide in training and validation
###################################
# Define the split ratio
train_ratio = 0.8  # 80% for training, 20% for validation

# Shuffle the dataframe (optional but recommended)
dataset = dataset.sample(frac=1).reset_index(drop=True)

# Initialize an empty column for set labels
dataset['Train_test'] = np.nan

# Group by label and assign training/validation sets
for label in dataset['essence_cat'].unique():
    # Get the indices for this label
    label_indices = dataset[dataset['essence_cat'] == label].index
    # Determine how many should go into validation
    n_validation = int(len(label_indices) * (1 - train_ratio))
    
    # Randomly assign the first n_validation samples to the validation set
    validation_indices = np.random.choice(label_indices, n_validation, replace=False)
    dataset.loc[validation_indices, 'Train_test'] = 1
    
    # The remaining go to the training set
    dataset['Train_test'].fillna(0, inplace=True)

dataset.to_csv("dataset_with_train_test.csv")
 
#Over or undersample
###################################
if opt.undersample:
    print("-----Under/oversampling the dataset-----")
    dataset = resample(dataset, sampling= "over", num_classes=2, NearMissV = 3, seed=2)
    print("-----Saving the undersampled dataset-----")
    dataset.to_csv("post_undersample_check.csv", index=False)
    print("----Number of samples after under/oversamling-----")
    ##### Count the number of rows with value 1
    num_rows_with_1 = (dataset['essence_cat'] == 1).sum()
    ##### Count the number of rows with value 0
    num_rows_with_0 = (dataset['essence_cat'] == 0).sum()

    print("Number of rows with value 1:", num_rows_with_1)
    print("Number of rows with value 0:", num_rows_with_0)

    w0 = 1/(num_rows_with_0/(num_rows_with_0+num_rows_with_1))
    w1 = 1/(num_rows_with_1/(num_rows_with_0+num_rows_with_1))
    w0_norm = (w0 / (w0+w1))*2
    w1_norm = (w1 / (w0+w1))*2

    
DOY = DOY.repeat(len(dataset), 1, 1)
    

    
print(f"dataset shape: {dataset.shape}")
num_continuous = (dataset.shape[1]-3) * 4
print(f"number of continous variables: {num_continuous}")

##### Count the number of rows with value 1
num_rows_with_1 = (dataset['essence_cat'] == 1).sum()

##### Count the number of rows with value 0
num_rows_with_0 = (dataset['essence_cat'] == 0).sum()

print("Number of rows with value 1:", num_rows_with_1)
print("Number of rows with value 0:", num_rows_with_0)

#### Defining the weights simply as the inverse frequency of each class and rescaling to the number of classes
w0 = 1/(num_rows_with_0/(num_rows_with_0+num_rows_with_1))
w1 = 1/(num_rows_with_1/(num_rows_with_0+num_rows_with_1))
w0_norm = (w0 / (w0+w1))*2
w1_norm = (w1 / (w0+w1))*2
  

print('---- Initializing the dataloaders ----')

cat_dims, cat_idxs, con_idxs, X_train, y_train, ids_train, X_valid, y_valid, ids_valid, X_test, y_test, ids_test, train_mean, train_std, DOY_train, DOY_valid = data_prep_premade(ds_id=dataset, DOY = DOY, seed = opt.dset_seed, task=opt.task)
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


train_ds = DataSetCatCon(X_train, y_train, DOY_train, ids_train, cat_idxs, opt.dtask)#, continuous_mean_std)
trainloader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True,num_workers=1)

valid_ds = DataSetCatCon(X_valid, y_valid, DOY_valid, ids_valid, cat_idxs, opt.dtask)#, continuous_mean_std)
validloader = DataLoader(valid_ds, batch_size=opt.batchsize, shuffle=False,num_workers=1)

test_ds = DataSetCatCon(X_test, y_test, DOY_valid, ids_test, cat_idxs, opt.dtask)#, continuous_mean_std)
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
y_dim = y_dim
)

vision_dset = opt.vision_dset
print(y_dim)
print(opt.task)

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
#model.load_state_dict(torch.load("/content/drive/MyDrive/Bee mapping spacetimeformer/Weights experiments/Brunswick_final2.pth"))


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
        ids, DOY, x_categ, x_cont, y_gts = data[0].to(device), data[1].to(device).type(torch.float32),data[2].to(device).type(torch.float32),data[3].to(device).type(torch.float32),data[4].type(torch.LongTensor).to(device)#,data[5].to(device).type(torch.float32)#,data[6].to(device).type(torch.float32)
        if opt.train_noise_type is not None and opt.train_noise_level>0:
            noise_dict = {
                'noise_type' : opt.train_noise_type,
                'lambda' : opt.train_noise_level
            }
            if opt.train_noise_type == 'cutmix':
                x_categ, x_cont = add_noise(x_categ,x_cont, noise_params = noise_dict)
        # We are converting the data to embeddings in the next step
        _ , x_categ_enc, x_cont_enc, con_mask = embed_data_mask(x_categ, x_cont, model,vision_dset,DOY=DOY)           
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
                        if test_kappa > best_kappa:
                            best_valid_accuracy = accuracy
                            best_test_auroc = test_auroc
                            best_test_accuracy = test_accuracy
                            torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))
                    else:
                        if test_kappa > best_kappa:
                            best_kappa = test_kappa
                            best_test_auroc = test_auroc
                            best_test_accuracy = test_accuracy               
                            torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))

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
                        torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))
            model.train()
                



print('APPLYING MODEL %s/bestmodel.pth' % (modelsave_path))
model.load_state_dict(torch.load('%s/bestmodel.pth' % (modelsave_path)))


# Make predictions
idxs_val, predictions_val, correct_val, ys_val, c0_prob_val, c1_prob_val = make_predictions(model, validloader, device)


#for the validation set
d = {'idx':idxs_val,'Prediction':predictions_val,'ys':ys_val,'correct':correct_val, 'c0_prob': c0_prob_val, 'c1_prob': c1_prob_val}
df = pd.DataFrame(data=d)
df['train_test'] = dataset["Train_test"]
# Save the predictions to a CSV file
df.to_csv("/content/drive/MyDrive/Bee mapping spacetimeformer/output_files/val_set_unbalanced"  + opt.output_name, index=False)

# Make predictions
idxs_val, predictions_val, correct_val, ys_val, c0_prob_val, c1_prob_val = make_predictions(model, trainloader, device)


#for the validation set
d = {'idx':idxs_val,'Prediction':predictions_val,'ys':ys_val,'correct':correct_val, 'c0_prob': c0_prob_val, 'c1_prob': c1_prob_val}
df = pd.DataFrame(data=d)
df['train_test'] = dataset["Train_test"]
# Save the predictions to a CSV file
df.to_csv("/content/drive/MyDrive/Bee mapping spacetimeformer/output_files/train_set_unbalanced"  + opt.output_name, index=False)


if opt.apply_version:
        
    # List the files in the directory
    folder_path = "/content/drive/MyDrive/Bee mapping spacetimeformer/saint + spacetimeformer/datasets_apply_brunswick"
    files = os.listdir(folder_path)
    
    # Filter files that match your criteria (assuming you only want to load files that contain 'Brunswick_cleaned_polys')
    files = [file for file in files if "Brunswick_cleaned_apply_polys_larger_than_8" in file]
    
    # Sort the files based on the fifth character in the filename
    files_sorted = sorted(files, key=lambda x: int(x[4]))  # Sorting by the fifth character (index 4) converted to an integer
    print(files_sorted)
    
    # Initialize a list to store DataFrames
    dfs = []
    
    # Load the sorted files into the list of DataFrames
    for file in files_sorted:
        df = pd.read_csv(os.path.join(folder_path, file))
        df = df.dropna()  # Drop rows with NaN values if needed
        dfs.append(df)
        
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
    
    
    print(f"dataset shape: {dataset.shape}")
    num_continuous = (dataset.shape[1]-3) * 4
    print(f"number of continous variables: {num_continuous}")
    dataset = dataset.reset_index(drop=True)
    cat_dims_pre, cat_idxs_pre, con_idxs_pre, X_train_pre, y_train_pre, ids_train_pre, X_valid_pre, y_valid_pre, ids_valid_pre, X_test_pre, y_test_pre, ids_test_pre, train_mean_pre, train_std_pre, DOY_train_pre, DOY_valid_pre = data_prep_premade(ds_id=dataset, DOY = DOY, seed = opt.dset_seed, task=opt.task)
    print(X_train_pre['data'].shape)
    #continuous_mean_std = np.array([train_mean,train_std]).astype(np.float32) 
    ds = DataSetCatCon(X_train_pre, y_train_pre, DOY_train_pre, ids_train_pre, cat_idxs_pre, opt.dtask)#, continuous_mean_std=continuous_mean_std)
    
    predictloader = DataLoader(ds, batch_size=opt.batchsize, shuffle=True,num_workers=1)
    
    #model_scripted = torch.jit.script(model) # Export to TorchScript
    #model_scripted.save('model_scripted.pt')
    
    # Make predictions
    idxs, predictions, correct, ys, c0_prob, c1_prob = make_predictions(model, predictloader, device)
    
    # Create a DataFrame with the predictions
    print(f"length idxs_val: {len(idxs)}")
    print(f"length c0_prob_val: {len(c0_prob)}")
    d = {'idx':idxs,'Prediction':predictions,'ys':ys,'correct':correct, 'c0_prob': c0_prob, 'c1_prob': c1_prob}
    df = pd.DataFrame(data=d)
    df['train_test'] = dataset["Train_test"]
    # Save the predictions to a CSV file
    df.to_csv("/content/drive/MyDrive/Bee mapping spacetimeformer/output_files/apply_subset" + opt.output_name, index=False)
    


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
