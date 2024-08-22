# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 09:42:52 2023

@author: Robbe Neyns
"""

import torch
from sklearn.metrics import roc_auc_score, mean_squared_error,confusion_matrix
import numpy as np
from augmentations_extra_embedding import embed_data_mask
import torch.nn as nn
import sklearn

def make_default_mask(x):
    mask = np.ones_like(x)
    mask[:,-1] = 0
    return mask

def tag_gen(tag,y):
    return np.repeat(tag,len(y['data']))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  

def get_scheduler(args, optimizer):
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    elif args.scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                      milestones=[args.epochs // 2.667, args.epochs // 1.6, args.epochs // 1.142], gamma=0.1)
    return scheduler

def imputations_acc_justy(model,dloader,device):
    model.eval()
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            ids =                 data[0].to(device) 
            DOY =                 data[1].to(device).type(torch.float32)
            satellite_azimuth =   data[2].to(device).type(torch.float32)
            sun_azimuth =         data[3].to(device).type(torch.float32)
            sun_elevation =       data[4].to(device).type(torch.float32)
            view_angle =          data[5].to(device).type(torch.float32)
            x_categ =             data[6].to(device).type(torch.float32)
            x_cont =              data[7].to(device).type(torch.float32)
            y_gts =               data[8].to(device).type(torch.LongTensor)
            # We are converting the data to embeddings in the next step
            _ , x_categ_enc, x_cont_enc, con_mask = embed_data_mask(x_categ, x_cont, model,vision_dset, DOY=DOY, satellite_azimuth=satellite_azimuth, sun_azimuth=sun_azimuth, sun_elevation=sun_elevation, view_angle=view_angle)           
            reps = model.transformer(x_categ_enc, x_cont_enc, con_mask)
            # select only the representations corresponding to y and apply mlp on it in the next step to get the predictions.
            y_reps = reps[:,0,:]
            y_outs = model.mlpfory(y_reps).to(device)
            # import ipdb; ipdb.set_trace()   
            y_test = torch.cat([y_test,x_categ[:,-1].float()],dim=0)
            y_pred = torch.cat([y_pred,torch.argmax(m(y_outs), dim=1).float()],dim=0)
            prob = torch.cat([prob,m(y_outs)[:,-1].float()],dim=0)
     
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]*100
    auc = roc_auc_score(y_score=prob.cpu(), y_true=y_test.cpu())
    return acc, auc


def multiclass_acc_justy(model,dloader,device):
    model.eval()
    vision_dset = True
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            ids =                 data[0].to(device) 
            DOY =                 data[1].to(device).type(torch.float32)
            satellite_azimuth =   data[2].to(device).type(torch.float32)
            sun_azimuth =         data[3].to(device).type(torch.float32)
            sun_elevation =       data[4].to(device).type(torch.float32)
            view_angle =          data[5].to(device).type(torch.float32)
            x_categ =             data[6].to(device).type(torch.float32)
            x_cont =              data[7].to(device).type(torch.float32)
            y_gts =               data[8].to(device).type(torch.LongTensor)
            # We are converting the data to embeddings in the next step
            _ , x_categ_enc, x_cont_enc, con_mask = embed_data_mask(x_categ, x_cont, model,vision_dset, DOY=DOY, satellite_azimuth=satellite_azimuth, sun_azimuth=sun_azimuth, sun_elevation=sun_elevation, view_angle=view_angle)           
            reps = model.transformer(x_categ_enc, x_cont_enc, con_mask)
            # select only the representations corresponding to y and apply mlp on it in the next step to get the predictions.
            y_reps = reps[:,0,:]
            y_outs = model.mlpfory(y_reps).to(device)
            # import ipdb; ipdb.set_trace()   
            y_test = torch.cat([y_test,x_categ[:,-1].float()],dim=0)
            y_pred = torch.cat([y_pred,torch.argmax(m(y_outs), dim=1).float()],dim=0)
     
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]*100
    return acc, 0

def class_wise_acc(y_pred,y_test,num_classes):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1) 
    correct_pred = (y_pred_tags == y_test).float()
    acc_classwise = []
    total_correct = []
    total_val_batch = []
    
    for i in range(num_classes):
        correct_pred_class = correct_pred[y_test == i]
        acc_class = correct_pred_class.sum() / len(correct_pred_class)
        acc_classwise.append(acc_class)
        total_correct.append(correct_pred_class.sum().cpu().data.numpy())
        #total_val_batch.append(torch.tensor(len(correct_pred_class), dtype=torch.int8))
        total_val_batch.append(len(correct_pred_class))
        #print('Accuracy of {} : {} / {} = {:.4f} %'.format(i, correct_pred_class.sum() , len(correct_pred_class) , 100 * acc_class))

    return acc_classwise, total_correct, total_val_batch 

def class_wise_acc_(model,dloader,device):
    model.eval()
    vision_dset = False
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)

    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            ids =                 data[0].to(device) 
            DOY =                 data[1].to(device).type(torch.float32)
            satellite_azimuth =   data[2].to(device).type(torch.float32)
            sun_azimuth =         data[3].to(device).type(torch.float32)
            sun_elevation =       data[4].to(device).type(torch.float32)
            view_angle =          data[5].to(device).type(torch.float32)
            x_categ =             data[6].to(device).type(torch.float32)
            x_cont =              data[7].to(device).type(torch.float32)
            y_gts =               data[8].to(device).type(torch.LongTensor)
            # We are converting the data to embeddings in the next step
            _ , x_categ_enc, x_cont_enc, con_mask = embed_data_mask(x_categ, x_cont, model,vision_dset, DOY=DOY, satellite_azimuth=satellite_azimuth, sun_azimuth=sun_azimuth, sun_elevation=sun_elevation, view_angle=view_angle)           
            reps = model.transformer(x_categ_enc, x_cont_enc, con_mask)
            # select only the representations corresponding to y and apply mlp on it in the next step to get the predictions.
            y_reps = reps[:,0,:]
            y_outs = model.mlpfory(y_reps).to(device)
            # import ipdb; ipdb.set_trace()   
            y_test = torch.cat([y_test.to(device),y_gts.to(device)],dim=0)
            y_pred = torch.cat([y_pred.to(device),y_outs.to(device)],dim=0)

    acc_classwise, total_correct, total_val_batch = class_wise_acc(y_pred,y_test,num_classes=5)
    
    # Compute the confusion matrix
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
    conf_matrix = confusion_matrix(y_test.cpu().numpy(),
                                   y_pred_tags.cpu().numpy())
    return acc_classwise, conf_matrix


def classification_scores(model, dloader, device, task,vision_dset):
    model.eval()
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            ids =                 data[0].to(device) 
            DOY =                 data[1].to(device).type(torch.float32)
            satellite_azimuth =   data[2].to(device).type(torch.float32)
            sun_azimuth =         data[3].to(device).type(torch.float32)
            sun_elevation =       data[4].to(device).type(torch.float32)
            view_angle =          data[5].to(device).type(torch.float32)
            x_categ =             data[6].to(device).type(torch.float32)
            x_cont =              data[7].to(device).type(torch.float32)
            y_gts =               data[8].to(device).type(torch.LongTensor)
            # We are converting the data to embeddings in the next step
            _ , x_categ_enc, x_cont_enc, con_mask = embed_data_mask(x_categ, x_cont, model,vision_dset, DOY=DOY, satellite_azimuth=satellite_azimuth, sun_azimuth=sun_azimuth, sun_elevation=sun_elevation, view_angle=view_angle)           
            reps = model.transformer(x_categ_enc, x_cont_enc, con_mask)
            # select only the representations corresponding to y and apply mlp on it in the next step to get the predictions.
            y_reps = reps[:,0,:]
            y_outs = model.mlpfory(y_reps)
            # import ipdb; ipdb.set_trace()   
            y_test = torch.cat([y_test.to(device),y_gts.squeeze().float().to(device)],dim=0)
            y_pred = torch.cat([y_pred.to(device),torch.argmax(y_outs, dim=1).float().to(device)],dim=0)
            if task == 'binary':
                prob = torch.cat([prob,m(y_outs)[:,-1].float()],dim=0)
     
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]*100
    kappa = sklearn.metrics.cohen_kappa_score(np.array(y_pred.cpu()),np.array(y_test.cpu()))
    auc = 0
    if task == 'binary':
        auc = roc_auc_score(y_score=prob.cpu(), y_true=y_test.cpu())
    return acc.cpu().numpy(), auc, kappa

def mean_sq_error(model, dloader, device, vision_dset):
    model.eval()
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            ids =                 data[0].to(device) 
            DOY =                 data[1].to(device).type(torch.float32)
            satellite_azimuth =   data[2].to(device).type(torch.float32)
            sun_azimuth =         data[3].to(device).type(torch.float32)
            sun_elevation =       data[4].to(device).type(torch.float32)
            view_angle =          data[5].to(device).type(torch.float32)
            x_categ =             data[6].to(device).type(torch.float32)
            x_cont =              data[7].to(device).type(torch.float32)
            y_gts =               data[8].to(device).type(torch.LongTensor)
            # We are converting the data to embeddings in the next step
            _ , x_categ_enc, x_cont_enc, con_mask = embed_data_mask(x_categ, x_cont, model,vision_dset, DOY=DOY, satellite_azimuth=satellite_azimuth, sun_azimuth=sun_azimuth, sun_elevation=sun_elevation, view_angle=view_angle)           
            reps = model.transformer(x_categ_enc, x_cont_enc, con_mask)
            # select only the representations corresponding to y and apply mlp on it in the next step to get the predictions.
            y_reps = reps[:,0,:]
            y_outs = model.mlpfory(y_reps)
            y_test = torch.cat([y_test,y_gts.squeeze().float()],dim=0)
            y_pred = torch.cat([y_pred,y_outs],dim=0)
        # import ipdb; ipdb.set_trace() 
        rmse = mean_squared_error(y_test.cpu(), y_pred.cpu(), squared=False)
        return rmse