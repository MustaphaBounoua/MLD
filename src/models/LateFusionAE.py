import numpy as np
import torch
import torch.nn as nn
from src.abstract.multimodal import MG
from src.utils import concat_vect , deconcat
MODEL_STR = "LateFusionAE"
from src.components.sde import VP_SDE
from src.utils import get_stat
import os
import pickle

class LateFusionAE(MG):
    """ 
    
    Late fusion of the latent space 
    
 
    
    """

    def __init__(self, aes,
                 train_loader=None,
                 test_loader=None,
                 model_name=MODEL_STR, 
      
                 ):
        self.modalities_list = [ae.modality for ae in aes]
        self.encoders = [ae.encoder for ae in aes]
        self.decoders = [ae.decoder for ae in aes]
        self.latent_dim = sum([mod.latent_dim for mod in self.modalities_list])
        
        super(LateFusionAE, self).__init__(
                latent_dim =self.latent_dim, 
                modalities_list=   self.modalities_list, 
                test_loader=test_loader,
                train_loader=train_loader,
                model_name=model_name,
                do_class=False,
                do_fd= False,
                eval_epoch=0,
                do_evaluation=True,
                nb_batchs=None,
                train_batch_size=512,
                n_fd=5000,
                subsampling_strategy="fullset"
                )
        self.sde = VP_SDE(liklihood_weighting=False,
                          device = self.device ,
                          beta_min= 0.1, 
                           beta_max= 20, 
                           importance_sampling=False)

     
    def encode(self, x):
        
        encodings = {}
        for idx, modality in enumerate(self.modalities_list):
            if modality.name in x.keys():
                self.encoders[idx].eval()
                mod_data = x[modality.name]
                latent = self.encoders[idx](mod_data)
                encodings[modality.name] = latent
        return encodings


    def decode(self, z):
        decodings = {}
        for idx, modality in enumerate(self.modalities_list):
            self.decoders[idx].eval()
            decodings[modality.name] = self.decoders[idx](z[modality.name])
 
        return decodings

    def forward(self,x):
        
        encodings = self.encode(x)
        z_mods = encodings   
                
        reconstruction = self.decode(z_mods)

        return reconstruction, z_mods
    
    
    
    ## Used to reproduce results in Figure A.1
    def compute_loss(self, x):
        
        self.sde.device = self.device
       
        self.eval()
        with torch.no_grad():
            encodings = self.encode(x)
        
        self.stat = {
            key:{
                "mean": encodings[key].mean(dim=0).detach(),
                "std": encodings[key].std(dim=0).detach(),
                "min": encodings[key].min().detach(),
                "max": encodings[key].max().detach()
            }
                for key in x.keys()
            
        }
        
        
        t_index = [0.0,0.05,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        #t_index = [-1,0,0.1]
        result_list = dict()
        for t in t_index:
           # print(self.stat)
            self.t = t
            results = self.evaluation()
            result_list[t] = results
            print("okk")
            print(results)
            
        with open(os.path.join(self.logger.log_dir,"results.pickle"),"wb") as f:
                        pickle.dump(result_list,f)
                
           # mnist_accr = results["Coherence"]["mnist"]["mnist,svhn"]
           # svhn_accr = results["Coherence"]["svhn"]["mnist,svhn"]
           # svhn_fid = results["FID"]["svhn"]["mnist,svhn"]
           # mnist_fid = results["FID"]["mnist"]["mnist,svhn"]
            
           # if t == -1:
           #     self.logger.experiment.add_scalars("accuracy of dataset", {"mnist":mnist_accr,"svhn":svhn_accr}, 0)
           #     self.logger.experiment.add_scalars("Fid of dataset ", {"mnist":mnist_fid,"svhn":svhn_fid}, t*1000)
           # else:
           #     self.logger.experiment.add_scalars("accuracy of ae ", {"mnist":mnist_accr,"svhn":svhn_accr}, t*1000)
           #     self.logger.experiment.add_scalars("Fid of ae ", {"mnist":mnist_fid,"svhn":svhn_fid}, t*1000)
        return {"loss" : 0.0}
    
    

    def compute_reconstruction_error(self, x, reconstruction, batch_size):
        recons_log = {}
        
        logprobs = torch.zeros(len(x)).type_as(x[self.modalities_list[0].name])
        #weights = torch.zeros(len(x)).type_as(x[self.modalities_list[0].name])
        for  idx, mod in enumerate( self.modalities_list ):
            logprobs[idx] = ( - mod.calc_log_prob( x[mod.name], reconstruction[mod.name] ) / batch_size )
            #weights[idx] = float(mod.reconstruction_weight)  
            recons_log[mod.name] = logprobs[idx]
        return  { "total": (logprobs).sum(dim=0), "rec_loss": recons_log}
    
    
    def sample(self, N):
        with torch.no_grad():
            z_mods = {}
            for mod in self.modalities_list:
                z_mods[mod.name] =  torch.randn(N,  mod.latent_dim, device=self.device)
            decodings = self.decode(z_mods)
            return decodings


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001 , betas=(0.9,0.999), weight_decay = 0 )
        return optimizer
   
   
    def conditional_gen_all_subsets(self, x,N=None):
        self.eval()
        results = {}
        modalities_str = np.array([mod.name for mod in self.modalities_list])
        subsets = { ','.join(modalities_str[s]) : s for s in self.subset_list}
            
        with torch.no_grad():
            encodings = self.encode(x)
            for idx, s_key in enumerate(subsets):
                sub_encodings = {
                    modalities_str[mod_i] : encodings[modalities_str[mod_i]]   for mod_i in subsets[s_key]
                }
              
                z_mods = sub_encodings   
                
                #posterior = self.posterior(z_mods)
                if self.t ==-1:
                    #z_mods = self.perturb_latent(z_mods,self.t)
                    #reconstruction = self.decode(z_mods)
                    results[s_key] = x
                elif self.t > 1:
                    for key in z_mods.keys():
                        z_mods[key]  = torch.randn_like(z_mods[key])
                    reconstruction = self.decode(z_mods)
                    results[s_key] = reconstruction
                else:
                    z_mods = self.standerdize(z_mods)
                    z_mods = self.perturb_latent(z_mods,self.t)
                    z_mods = self.destanderdize(z_mods)
                    reconstruction = self.decode(z_mods)
                    results[s_key] = reconstruction
               # results[s_key] = x
        return results 
    
    
    
    def standerdize(self, encodings):
        for key in encodings.keys(): 
                    encodings[key] = (encodings[key] - self.stat[key]["mean"] ) /(self.stat[key]["std"] ) 
        return encodings
     
     
    def destanderdize(self, encodings):
        for key in encodings.keys():
                    encodings[key] = (encodings[key]   * (self.stat[key]["std"] ) )  + self.stat[key]["mean"]
        return encodings
    
    

    def denormalize(self, encodings):
       
        for key in encodings.keys():
            encodings[key] = (encodings[key]   * (self.stat[key]["max"] - self.stat[key]["min"] ) )  + self.stat[key]["min"]
        return encodings

    def normalize(self, encodings):
        for key in encodings.keys():
            encodings[key] = (encodings[key] - self.stat[key]["min"] ) /(self.stat[key]["max"] - self.stat[key]["min"] ) 
        return encodings
    
 




    def perturb_latent(self,z_mods,t):
        z_concat = concat_vect(z_mods)

        time  = t * torch.ones(z_concat.size(0),1).to(self.device)
            ## random z ## integral of the brownian noise
        noise = torch.randn_like(z_concat).to(self.device)
            ## Estimate P_t(x)
            ## integral of  drift and diffusion coefficient 

        mean, std = self.sde.marg_prob(time,z_concat)
           
        z_concat = z_concat * mean + std * noise    
        return deconcat(z_concat,self.modalities_list)

    def perturb_latent(self,z_mods,t):
        z_concat = concat_vect(z_mods)

        time  = t * torch.ones(z_concat.size(0),1).to(self.device)
            ## random z ## integral of the brownian noise
        noise = torch.randn_like(z_concat).to(self.device)
            ## Estimate P_t(x)
            ## integral of  drift and diffusion coefficient 

        mean, std = self.sde.marg_prob(time,z_concat)
           
        z_concat = z_concat * mean + std * noise    
        return deconcat(z_concat,self.modalities_list)

    # def perturb_latent(self,z_mods,t):
    #     z_concat = concat_vect(z_mods)

    #     for key in z_mods.keys():
    #         z_mod = z_mods[key]
    #         time  = t * torch.ones(z_mod.size(0),1).to(self.device)
    #         ## random z ## integral of the brownian noise
    #         noise = torch.randn_like(z_mod).to(self.device)
    #         ## Estimate P_t(x)
    #         ## integral of  drift and diffusion coefficient 

    #         mean, std = self.sde.marg_prob(time,z_mod)
           
    #         z_mods[key] = z_mod * mean + std * noise    
    #     return deconcat(z_concat,self.modalities_list)
    
    
    
    
    
    
    
    def conditional_gen_latent_subsets(self, x):
        
        results = {}
        modalities_str = np.array([mod.name for mod in self.modalities_list])
        subsets = { ','.join(modalities_str[s]) : s for s in self.subset_list}
            
        with torch.no_grad():
            encodings = self.encode(x)
            for idx, s_key in enumerate(subsets):
                sub_encodings = {
                    modalities_str[mod_i] : encodings[modalities_str[mod_i]]   for mod_i in subsets[s_key]
                }
                
                z_mods = sub_encodings   
                
                posterior = self.posterior(z_mods)
                #reconstruction = self.decode(posterior)
                
                results[s_key] = [concat_vect(posterior)]
               
        return results 
    
    