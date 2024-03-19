import sys
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from tqdm import tqdm
from sklearn.manifold import TSNE
import plotly.express as px


def train_clf_lr_all_subsets(model,subsets_dict , d_loader ,batch_size, class_dim, device,num_training_samples_lr):
    mm_vae = model
    mm_vae.eval()
    subsets = subsets_dict
    
    num_batches_epoch = int(d_loader.dataset.__len__() /float(batch_size));
    
    class_dim = class_dim
    
    #n_samples = int(d_loader.dataset.__len__())
    
    n_samples = batch_size*20
    
    data_train = dict();
    for k, s_key in enumerate(subsets.keys()):
        if s_key != '':
            data_train[s_key] = np.zeros((n_samples,
                                          class_dim))
    
    all_labels = np.zeros((n_samples, 1))
    
    for it, batch in enumerate(d_loader):
        if ( it+1 > 20):
            break;
        
        batch_d = batch[0]
        batch_l = batch[1]
        #print(batch_size)
        
        for k, m_key in enumerate(batch_d.keys()):
            
            batch_d[m_key] = batch_d[m_key].to(device)
            
   
        lr_subsets = mm_vae.conditional_gen_latent_subsets(batch_d)
   
        all_labels[(it*batch_size):((it+1)*batch_size), :] = np.reshape(batch_l, (batch_size,1))
    
        for k, key in enumerate(lr_subsets.keys()):
            data_train[key][(it*batch_size):((it+1)*batch_size), :] = lr_subsets[key][0].cpu().data.numpy();

    n_train_samples = num_training_samples_lr
    rand_ind_train = np.random.randint(n_samples, size=n_train_samples)
    labels = all_labels[rand_ind_train,:]
    
  
    
    for k, s_key in enumerate(subsets.keys()):
        if s_key != '':
            d = data_train[s_key];
            data_train[s_key] = d[rand_ind_train, :]
            
    clf_lr = train_clf_lr( data_train, labels )
    return clf_lr;
 

def test_clf_lr_all_subsets( clf_lr, model,subsets,d_loader,batch_size,device, nb_batchs = None ):
    mm_vae = model
    mm_vae.eval()
    subsets = subsets
  

    lr_eval = dict()
    for k, s_key in enumerate(subsets.keys()):
        if s_key != '':
            lr_eval[s_key] = []



    num_batches_epoch = int(d_loader.__len__() /float(batch_size))
    
    for iteration, batch in tqdm ( enumerate(d_loader), desc=" Evaluating the latent space quality via linear clasification" ):
      #  if  nb_batchs != None and iteration > nb_batchs :
      #      break
        batch_d = batch[0]
        batch_l = batch[1]
        
        
        for k, m_key in enumerate(batch_d.keys()):
            batch_d[m_key] = batch_d[m_key].to(device);
            

        lr_subsets= mm_vae.conditional_gen_latent_subsets(batch_d)
        
        data_test = dict()
        
        for k, key in enumerate(lr_subsets.keys()):
            data_test[key] = lr_subsets[key][0].cpu().data.numpy()
        evals = classify_latent_representations(clf_lr,
                                                data_test,
                                                batch_l)
    
        eval_label = evals
        for k, s_key in enumerate(eval_label.keys()):
            lr_eval[s_key].append(eval_label[s_key])
                
                
    for l, l_key in enumerate(lr_eval.keys()):
        lr_eval[l_key] = mean_eval_metric(lr_eval[l_key]);
    return lr_eval;



def mean_eval_metric( values):
        return np.mean(np.array(values));
    
    

def classify_latent_representations(clf_lr, data, labels):
    
    labels = np.array(np.reshape(labels, (labels.shape[0], 1)));
    
    eval_all_labels = dict()
    gt = labels
    clf_lr_label = clf_lr
    eval_all_reps = dict()
    
    for s_key in data.keys():
        data_rep = data[s_key]
        clf_lr_rep = clf_lr_label[s_key]
        y_pred_rep = clf_lr_rep.predict(data_rep)
        eval_label_rep = accuracy_score(gt.ravel(),
                                             y_pred_rep.ravel())
        eval_all_reps[s_key] = eval_label_rep
    eval_all_labels = eval_all_reps
    return eval_all_labels;


def train_clf_lr( data, labels):
    
    labels = np.reshape(labels, (labels.shape[0], 1))
    gt = labels;
    
    clf_lr_reps = dict();
    for s_key in data.keys():
        data_rep = data[s_key];
        clf_lr_s = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto', max_iter=1000);
        clf_lr_s.fit(data_rep, gt.ravel());
        clf_lr_reps[s_key] = clf_lr_s
    return clf_lr_reps




def tsne_plot(z_latent,labels):

    latent_dim = z_latent.size(-1)
    df = pd.DataFrame(z_latent.cpu().numpy())
    df["label"] = labels.cpu().numpy()
    df["label"] = df['label'].apply(str)
    features = df.loc[:,np.arange(latent_dim)]
    tsne = TSNE(n_components=2, random_state=0)
    projections = tsne.fit_transform(features)
    fig = px.scatter(
        projections, x=0, y=1,
        color=df.label, labels={'color': 'label'})
    return fig






def plot_latent_space(model, test_loader, modalities_list,filename,subset_list, device = "cuda", nb_samples = 5000, batch_size =256 ):
    
    model = model.to(device)
    z_list_train  = { str(subset) :torch.Tensor().to(device) for subset in subset_list} 
    ground_truth_train = torch.Tensor().to(device)

    nb_batch = (nb_samples // batch_size ) + 1
    for batch_idx, batch in enumerate(test_loader):
        
        if batch_idx == nb_batch:
            break
            
        data = batch[0]
        labels = batch[1]
        
        x = batch[0]
        data = [None] * len(modalities_list)
        
        for idx,_ in enumerate( modalities_list):
                data[idx] = x[idx].to(device)
                
        z_list = model.conditional_latent_all_subsets(data)
        
        
        for subset in subset_list:
            z_list_train[str(subset)] = torch.cat([z_list_train[str(subset)] ,z_list[str(subset)].to(device)])
        ground_truth_train = torch.cat([ground_truth_train,labels.to(device)])
    
    ground_truth = ground_truth_train
    for subset in subset_list:
        z_list = z_list_train[str(subset)]
        fig = tsne_plot(z_list[:nb_samples],ground_truth[:nb_samples])
        fig.write_image(filename+str(subset)+".png")
    