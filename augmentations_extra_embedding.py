# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 11:25:15 2023

@author: Robbe Neyns
"""

import torch
import numpy as np


def embed_data_mask(x_categ, x_cont,model,vision_dset=False,DOY=0, satellite_azimuth=0, sun_azimuth=0, sun_elevation=0, view_angle=0):
    device = x_cont.device
    #x_categ = x_categ + model.categories_offset.type_as(x_categ)
    #x_categ_enc = model.embeds(x_categ)
    x_categ_enc = x_categ ## just a little hack fix
    batch, n1,n2 = x_cont.shape
    #_, n3 = x_categ.shape
    if model.cont_embeddings == 'MLP':
        x_cont_enc = torch.empty(n1,n2, model.dim)
        for i in range(model.num_continuous):
            x_cont_enc[:,i,:] = model.simple_MLP[i](x_cont[:,i])
    elif model.cont_embeddings == 'temporal' or model.cont_embeddings == 'spatio-temporal':
        #x = torch.cat((DOY, satellite_azimuth, sun_azimuth, sun_elevation, view_angle), dim=-1)
        x = DOY
        x_cont_enc, _, _, con_mask = model.embedding(x_cont, x)
    else:
        raise Exception('This case should not work!')    



    x_cont_enc = x_cont_enc.to(device)

    #cat_mask_temp = cat_mask #+ model.cat_mask_offset.type_as(cat_mask)
    #print(f"con mask: {con_mask}")
    #print(f"con mask shape: {con_mask.shape} ")
    #print(f"con_mask_ofset shape: {model.con_mask_offset.type_as(con_mask).shape} ")
    con_mask_temp = con_mask #+ model.con_mask_offset.type_as(con_mask)

    #print(f"con_mask_temp after offset: {con_mask_temp.shape}")
    #cat_mask_temp = model.mask_embeds_cat(cat_mask_temp)
    con_mask_temp = model.mask_embeds_cont(con_mask_temp)
    #print(f"con_mask_temp after mask_embeds_cont: {con_mask_temp.shape}")
    #print(f"x_cont_enc shape: {x_cont_enc.shape}")
    #x_categ_enc[cat_mask == 0] = cat_mask_temp[cat_mask == 0] #it just fills the masked locations up? with what?
    #x_cont_enc[con_mask == 0] = con_mask_temp[con_mask == 0]

    if vision_dset:
        pos = np.tile(np.arange(x_categ.shape[-1]),(x_categ.shape[0],1))
        pos =  torch.from_numpy(pos).to(device)
        pos_enc =model.pos_encodings(pos)
        x_categ_enc+=pos_enc

    return x_categ, x_categ_enc, x_cont_enc, con_mask




def mixup_data(x1, x2 , lam=1.0, y= None, use_cuda=True):
    '''Returns mixed inputs, pairs of targets'''

    batch_size = x1.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)


    mixed_x1 = lam * x1 + (1 - lam) * x1[index, :]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index, :]
    if y is not None:
        y_a, y_b = y, y[index]
        return mixed_x1, mixed_x2, y_a, y_b
    
    return mixed_x1, mixed_x2


def add_noise(x_categ,x_cont, noise_params = {'noise_type' : ['cutmix'],'lambda' : 0.1}):
    lam = noise_params['lambda']
    device = x_categ.device
    batch_size = x_categ.size()[0]

    if 'cutmix' in noise_params['noise_type']:
        index = torch.randperm(batch_size)
        cat_corr = torch.from_numpy(np.random.choice(2,(x_categ.shape),p=[lam,1-lam])).to(device)
        con_corr = torch.from_numpy(np.random.choice(2,(x_cont.shape),p=[lam,1-lam])).to(device)
        x1, x2 =  x_categ[index,:], x_cont[index,:]
        x_categ_corr, x_cont_corr = x_categ.clone().detach() ,x_cont.clone().detach()
        x_categ_corr[cat_corr==0] = x1[cat_corr==0]
        x_cont_corr[con_corr==0] = x2[con_corr==0]
        return x_categ_corr, x_cont_corr
    elif noise_params['noise_type'] == 'missing':
        x_categ_mask = np.random.choice(2,(x_categ.shape),p=[lam,1-lam])
        x_cont_mask = np.random.choice(2,(x_cont.shape),p=[lam,1-lam])
        x_categ_mask = torch.from_numpy(x_categ_mask).to(device)
        x_cont_mask = torch.from_numpy(x_cont_mask).to(device)
        return torch.mul(x_categ,x_categ_mask), torch.mul(x_cont,x_cont_mask)
        
    else:
        print("yet to write this")