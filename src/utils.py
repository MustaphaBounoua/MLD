import os, shutil
import torch 
import sys
import pandas as pd
import numpy as np
pd.options.plotting.backend = "plotly"
from IPython.display import display, clear_output
from torchvision.utils import save_image,make_grid
import torchvision
from itertools import chain, combinations
import pickle
from src.dataLoaders.MHD.utils.trajectory_utils import *
import librosa
import torchaudio
import uuid
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os
from pathlib import Path
FOLDER = "/temp_fid/"
import math



def get_stat(file_name):
    t = pickle.load(open(file_name, "rb"))
    for key in t.keys():
        if  key!= 'cat':
            t[key]['mean'] = t[key]['mean'].to("cuda")
            t[key]['std'] = t[key]['std'].to("cuda")
    return t

def log_mean_exp(value, dim=0, keepdim=False):
    return torch.logsumexp(value, dim, keepdim=keepdim) - math.log(value.size(dim))


def cat_shared_private(encodings):
            return [
                torch.cat([encodings["private"][0] ,encodings["shared"][0] ],dim=-1  ), # mu
                torch.cat([encodings["private"][1] ,encodings["shared"][1] ],dim=-1  ) # logvar
            ]   


def get_root_folder(path_n):
    path = Path(path_n)
    return path.parent.absolute().parent.absolute()


dir_gen_eval_fid = "fid/"

def get_stat_from_file(file_name):
    t = pickle.load(open(file_name, "rb"))
    for key in t.keys():
        if  key!= 'cat':
            t[key]['mean'] = t[key]['mean'].to("cuda:0")
            t[key]['std'] = t[key]['std'].to("cuda:0")
    return t

def concat_vect(encodings):
    z = torch.Tensor()
    for key in encodings.keys():
        z = z.to(encodings[key].device)
        z = torch.cat( [z, encodings[key]],dim = -1 )
    return z 
    

def get_mask(modalities_list, subset, shape):
    mask = torch.zeros(shape)
    idx = 0
    for index_mod, mod in enumerate(modalities_list ):
        if index_mod in subset:
            mask[:, idx:idx + mod.latent_dim] = 1.0
        idx = idx + mod.latent_dim
    return mask


def stack_posterior(encodings):
  
    mu =torch.Tensor().type_as(encodings[list(encodings.keys())[0] ][0] )
    logvar=torch.Tensor().type_as(mu)
    
    for m_key in  encodings.keys():
        mu_mod,logvar_mod = encodings[m_key]
        
        mu = torch.cat((mu, mu_mod.view(1,mu_mod.size(0),mu_mod.size(1)   ) ), dim=0)
        logvar = torch.cat((logvar, logvar_mod.view(1,logvar_mod.size(0),logvar_mod.size(1)   ) ), dim=0)
        
    return mu,logvar


def concat_vect(encodings):
    z = torch.Tensor()
    for key in encodings.keys():
        z = z.to(encodings[key].device)
        z = torch.cat( [z, encodings[key]], dim = -1 )
    return z 

def deconcat(z,modalities_list):
    z_mods={}
    idx=0
    for mod in modalities_list:
        z_mods[mod.name] = z[:,idx:idx+mod.latent_dim]
        idx = idx+mod.latent_dim
    return z_mods


def stack_tensors(encodings):
  
    z_tensor = torch.Tensor().type_as(encodings[list(encodings.keys())[0] ] )
   
    
    for m_key in  encodings.keys():
        z = encodings[m_key]
        z_tensor = torch.cat((z_tensor, z.view(1,z.size(0),z.size(1)   ) ), dim=0)

    return z_tensor






def set_paths_fid(self, dir_fid, subsets):
    
    dir_real = os.path.join(dir_fid, 'real')
    dir_random = os.path.join(dir_fid, 'random')
    
    paths = {'real': dir_real,
                 'random': dir_random}
    
    dir_cond = dir_gen_eval_fid
    
    for k, name in enumerate(subsets):
            paths[name] = os.path.join(dir_cond, name)
            
    print(paths.keys())
    return paths;



def set_paths_fid(folder, subsets):
    
    dir_real = os.path.join(folder, 'real')
    dir_random = os.path.join(folder, 'random')
    
    create_forlder(folder)
    create_forlder(dir_real)
    create_forlder(dir_random)
    
    paths = {'real': dir_real,
                 'random': dir_random}
    
    dir_cond = os.path.join(folder, 'fid_eval')
    paths["dir_gen"]= folder
    create_forlder(dir_cond)
    
    for k, name in enumerate(subsets):
        paths[name] = os.path.join(dir_cond, name)
    
    return paths


def save_generated_samples_singlegroup(batch_id, group_name, samples, batch_size,modalities_list, paths_fid ):
    
    dir_save = paths_fid[group_name]
    
    for k, key in enumerate(samples.keys()):
        dir_f = os.path.join(dir_save, key)
        if not os.path.exists(dir_f):
            os.makedirs(dir_f)

    cnt_samples = batch_id * batch_size
    
    if batch_id ==0:
        for i, key in enumerate(samples.keys()):
            mod = modalities_list[i]
            if mod.fad != None:
                print("cleaning "+str(batch_id)+ os.path.join(dir_save, key) )
                clean_folder(os.path.join(dir_save, key))

    for k in range(0, batch_size):
        for i, key in enumerate(samples.keys()):
            data_mod = samples[key].cpu()
            mod = modalities_list[i]
            if mod.fad :

                fn_out = os.path.join(dir_save, key, str(cnt_samples).zfill(6) +
                                    mod.file_suffix)
                
                mod.save_data(data_mod[k], fn_out, {'img_per_row': 1})
        cnt_samples += 1


def save_one_trajectory(data, filename, traj_norm, tmp_path="temp/"):
 
    trajectory = data * (traj_norm['max'] - traj_norm['min']) + traj_norm['min']

    # Generate image of trajectory
    trajs = generate_image_from_trajectory(traj=trajectory.cpu(), tmp_path=tmp_path)[0]

    torchvision.utils.save_image(trajs,filename)
    return


def generate_image_from_trajectory(traj, tmp_path="temp/"):
    create_forlder(tmp_path)
    # Plot Trajectory in color and save image

    
    str_id = str(uuid.uuid4())

    fig, ax = plot_single_stroke_digit_evaluation(traj)
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(os.path.join(tmp_path,str_id+'tmp.png'), bbox_inches=extent, dpi=100)
    plt.close(fig)

    # Get image in greyscale
    g_img = get_greyscale_image(os.path.join(tmp_path,str_id+'tmp.png'))
    os.remove(os.path.join(tmp_path,str_id+'tmp.png'))
    np_img = np.asarray(g_img)
    return torch.from_numpy(np_img).unsqueeze(0).float()/float(255.), g_img  # Normalize data!



def get_trajectories_image(data, filename, traj_norm, tmp_path="temp/",nrow = 8):
    trajs = []
    create_forlder(tmp_path)
    for i in range(data.size(0)):
        # Unnormalize data
        trajectory = data[i] * (traj_norm['max'] - traj_norm['min']) + traj_norm['min']
        # Generate image of trajectory
        trajs.append(generate_image_from_trajectory(traj=trajectory.cpu(), tmp_path=tmp_path)[0])

    t_trajs = torch.stack(trajs, dim=0)
    grid = torchvision.utils.make_grid(t_trajs,
                                                                padding=5,
                                                                pad_value=.5,
                                                                nrow=nrow)
    if filename !="":
        torchvision.utils.save_image(grid,
                                    filename)
    return grid



def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
  fig, axs = plt.subplots(1, 1)
  axs.set_title(title or 'Spectrogram (db) - normalized')
  axs.set_ylabel(ylabel)
  axs.set_xlabel('frame')
  im = axs.imshow(spec.cpu().numpy(), origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  fig.colorbar(im, ax=axs)
  #plt.show()
  #plt.savefig("mygraph.png")
  return fig




def save_sound(data, filename, sound_norm):
    sound_list = []
    for i in range(data.size(0)):
        # Unormalize data
        wave = data[i].cpu() * (sound_norm['max'] - sound_norm['min']) + sound_norm['min']
        # Permute channels and remove channel
        wave = wave.permute(0, 2, 1).squeeze(0)
        # DB to Power
        wave = librosa.db_to_power(wave)
        # Generate wave using Griffin-Lim algorithm
        
        sound_wav = librosa.feature.inverse.mel_to_audio(wave.squeeze(0).data.numpy(),
                                                         sr=16000,
                                                         n_iter=60)
      
        # Save data
        audio = (torch.from_numpy(sound_wav) * np.iinfo(np.int16).max).view(1,-1)
        sound_list.append(audio)
        
        if filename != "":
            f_filename = filename + "_" + str(i) + ".wav"
            torchaudio.save(f_filename, audio , 16000)
    return sound_list




def save_wave_sound(data, filename, sound_norm):

    wave = data.cpu() * (sound_norm['max'] - sound_norm['min']) + sound_norm['min']
        # Permute channels and remove channel
    wave = wave.permute(0, 2, 1).squeeze(0)
        # DB to Power
    wave = librosa.db_to_power(wave)
        # Generate wave using Griffin-Lim algorithm
    sound_wav = librosa.feature.inverse.mel_to_audio(wave.squeeze(0).data.numpy(),
                                                         sr=16000,
                                                         n_iter=60)
    ## make to 16000 1s so fad can be computed
    sound_wav = F.pad(torch.from_numpy(sound_wav), (64,64), "constant", 0)
    # Save data
    audio = ( sound_wav * np.iinfo(np.int16).max).view(1,-1)
    
    if filename != "":
        torchaudio.save(filename, audio , 16000)
    return audio






def get_stat_(modalities_list , data_loader, ae_model, device):
    ae_model.to(device)
    ae_model.eval()
    nb_batch = 0

 
    stat = { key : {"max":-sys.maxsize , "min":sys.maxsize,"sum":0,"sum_squared":0  } for key in modalities_list}
    stat["cat"] = {"sum" : 0,"sum_squared":0,"max":-sys.maxsize , "min":sys.maxsize }
    
    for iteration, batch in tqdm( enumerate(data_loader), desc =" Executing encodings to compute latent_space stats" ):
            
            
            batch_d = batch[0]
            batch_l = batch[1]
            
            for k, m_key in enumerate(batch_d.keys()):
                batch_d[m_key] = batch_d[m_key].to(device);
                b_s = batch_d[m_key].size(0)
            encodings = ae_model.encode(batch_d)
            ##mnist:
    
            nb_batch += 1
        
            for key in encodings.keys():

                stat[key]["sum"] += torch.mean(encodings[key]).detach().cpu().numpy()
                stat[key]["sum_squared"] += torch.mean(encodings[key]**2).detach().cpu().numpy()
                
            
                
                themax = torch.max(encodings[key]).item()
                themin = torch.min(encodings[key]).item()
                
                if  themax>stat[key]["max"] :
                    stat[key]["max"] = themax
                if themin<  stat[key]["min"] :
                    stat[key]["min"] = themin
            z = concat_vect(encodings)  
            
            
            themax = torch.max(z).item()
            themin = torch.min(z).item()
                
            if  themax>stat["cat"]["max"] :
                    stat["cat"]["max"] = themax
            if themin<  stat["cat"]["min"] :
                    stat["cat"]["min"] = themin
            
            stat["cat"]["sum"] += torch.mean(z).detach().cpu().numpy()
            stat["cat"]["sum_squared"] += torch.mean(z**2).detach().cpu().numpy()
            
                    
    for key in modalities_list:
        stat[key]["mean"] = stat[key]["sum"] /nb_batch
        stat[key]["std"] = (stat[key]["sum_squared"]  / nb_batch - stat[key]["mean"] ** 2) ** 0.5
        
    stat["cat"]["mean"] = stat["cat"]["sum"] /nb_batch
    stat["cat"]["std"] = (stat["cat"]["sum_squared"]  / nb_batch - stat["cat"]["mean"] ** 2) ** 0.5
    
    
    print(stat)
    return stat


def set_subset(modalities, strategy = "powerset"):
    nb_mod =len(modalities)
    num_modalities = np.arange(0,nb_mod)
    if strategy == "powerset":
        subsets_list = chain.from_iterable(combinations(num_modalities, n) for n in range(1,nb_mod+1))
    elif strategy == "unimodal":
        subsets_list = chain.from_iterable(combinations(num_modalities, n) for n in range(1,2))
    elif strategy == "fullset":
         subsets_list = chain.from_iterable(combinations(num_modalities, n) for n in range(nb_mod,nb_mod+1))
    return [list(x) for x in subsets_list]


def concat_vect(encodings):
    z = torch.Tensor()
    for key in encodings.keys():
        z = z.to(encodings[key].device)
        z = torch.cat( [z, encodings[key]],dim = -1 )
    return z 

def write_samples_text_to_file(samples, filename):
    file_samples = open(filename, 'w');
    for k in range(0, len(samples)):
        file_samples.write(''.join(samples[k]) + '\n');
    file_samples.close();



def plot_grid(x,n_row):
    grid_img = make_grid(x.cpu(), nrow=n_row)
    img = torchvision.transforms.ToPILImage()(grid_img)
    img.show()
    plt.imshow(grid_img.permute(1, 2, 0).numpy().astype('uint8'))


def get_annealing_factor(epoch,epoch_annealing, batch_idx, n_batches):
    if epoch < epoch_annealing:
                # compute the KL annealing factor for the current mini-batch in the current epoch

                return  (float(batch_idx + (epoch - 1) * n_batches + 1) /
                                    float(epoch_annealing * n_batches))
    else:
                # by default the KL annealing factor is unity
                return 1.0



    





def clean_folder(folder):

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))






def draw_learning_curve( epoch,tracking_df ):

    clear_output(wait = True)
    fig= tracking_df.plot()
    fig.show()
   


def create_forlder(folder):

    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        print("Directory " , folder ,  " Created ")



def adjustmask(mask,input_size):
        mask=torch.tensor(mask).view(len(mask),1,1,1)
        return mask.expand(mask.size(0),*input_size)


def get_random_dropping_mask_2D(N,p):
        mask = np.random.choice(a=[1,2,3], size=(N), p=[p/2, p/2,1-p])
        mask_mat = np.array([mask ==1,mask ==2])
        return np.array(mask_mat ==0, dtype=int)



""" 
def get_random_dropping_index_2D(N,p):
        mask = np.random.choice(a=[1,2,3], size=(N), p=[p/2, p/2,1-p])
        mask_mat = np.array([mask ==1,mask ==2])
        index = np.arange(0,N)
        return np.ma.masked_array(index, mask_mat[0]).mask, np.ma.masked_array(index, mask_mat[1]).mask """
        



def save_pickle(file_path, data):
    
    f = open(file_path,"wb")
    pickle.dump(data,f)


def save_json(file_path,data):
    import json
    with open(file_path, 'w') as fp:
        json.dump(data, fp)
    
def read_pickle(file_path):
    a_file = open(file_path, "rb")
    return pickle.load(a_file)