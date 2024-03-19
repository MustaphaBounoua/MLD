
import torch
import torch.nn as nn
from src.abstract.multimodal import MG
from src.components.mm_sde_uni import VP_SDE
MODEL_STR = "Multimodal Latent Diffusion model"
from src.utils import get_stat 
from src.components.mlp_unet import UnetMLP


from src.utils import get_mask, concat_vect, deconcat
from copy import deepcopy
import os
import pickle



class EMA(nn.Module):
    """ Model Exponential Moving Average V2 from timm"""
    def __init__(self, model, decay=0.9999):
        super(EMA, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)



class MLD_Uni(MG):
    """ 
    
    Multimodal Latent Diffusion model with Unidiffuser training scheme
    
    
    """

    def __init__(self, aes,  train_loader,
                 test_loader,
                 model_name=MODEL_STR, 
                 nb_samples = 8, 
                 batch_size=256,
                 eval_epoch= 5,
                 do_evaluation=True,
                 do_fd = True,
                 log_epoch = 5,
                 d = 0.2,
                 n_fd = 5000,
                 lr = 0.001,
                 unet_type ="linear",
                 unet_architecture = (1,2,3,4),
                 group_norm = 8,
                 nb_batchs = 2,
                 init_dim= 128,
                 do_class = False,
                 ll_weight = False,
                 importance_sampling = True,
                 check_stat = True,
                 preprocess_op = "standardize",
                 preprocess_type ="modality",
                 train_batch_size =512,
                 time_dim = 96,
                 betas = [0.1,20.0],
                 N_step= 250, 
                 use_attention =True,
                 shift_scale = False,
                 debug = False,
                  dim_head = None,
                  num_head = 2,
                  cross_gen = "repaint",
                  use_ema = True,
                  dataset =None, limit_clip = 5000
                 ):
 
        super(MLD_Uni, self).__init__(
                model_name = model_name,
                latent_dim = aes.latent_dim, 
                modalities_list=   aes.modalities_list, 
                test_loader= test_loader,
                train_loader= train_loader,
                batch_size= batch_size,
                nb_samples = nb_samples, 
                eval_epoch = eval_epoch,
                do_evaluation= do_evaluation,
                do_fd = do_fd,
                log_epoch = log_epoch,
                n_fd = n_fd,
                lr = lr, limit_clip = limit_clip,
                nb_batchs = nb_batchs,
                do_class= do_class, 
                train_batch_size = train_batch_size,dataset= dataset
               
                )

        self.aes = aes
        self.modalities_list = aes.modalities_list
        self.latent_dim = aes.latent_dim
        self.encoders = None
        self.decoders = None
        self.do_check_stat = check_stat
        self.N_step = N_step
        self.preprocess_op = preprocess_op
        self.preprocess_type = preprocess_type
        self.d = d
        self.ll_weight = ll_weight
        self.importance_sampling= importance_sampling
        self.betas = betas
        self.use_attention = use_attention
        self.shift_scale = shift_scale
        self.debug = debug
        #self.save_hyperparameters(ignore= ["modalities_list","train_loader","test_loader","aes","betas"])
        
        if unet_type == "linear":
            self.score = UnetMLP( dim=self.latent_dim,
                                  time_dim = time_dim,  
                                  resnet_block_groups=group_norm, 
                                  dim_mults= unet_architecture , 
                                  shift_scale= shift_scale,
                                  use_attention= use_attention,
                                  init_dim = init_dim,
                                  nb_mod= len(self.modalities_list), num_head=num_head,
                                  dim_head = dim_head, modalities = self.modalities_list                                  
                                  )
        
        self.use_ema = use_ema
        self.model_ema = EMA(self.score, decay=0.999) if use_ema else None

        self.sde = VP_SDE(liklihood_weighting=ll_weight,
                          device = self.device ,
                          nb_mod= len(self.modalities_list),
                          beta_min= self.betas[0], 
                          beta_max= self.betas[1], 
                          importance_sampling=importance_sampling,
                          method = cross_gen
                          )
     
        self.mask_subset = []
        for idx,key  in enumerate(self.subset_list_dict.keys() ):
            self.mask_subset.append( { "subset":self.subset_list_dict[key],
                "mask": get_mask(modalities_list= self.modalities_list , 
                                             subset= self.subset_list_dict[key], 
                                             shape=(self.train_batch_size,self.latent_dim)) ,
                                    "time_mask" : torch.tensor([ 1 if i in self.subset_list_dict[key] else 0 for i in range(len(self.modalities_list))]).to(self.device).expand((self.batch_size,len(self.modalities_list) )) 
                                   }  )

        self.stat = None
        

    def on_before_backward(self, loss: torch.Tensor) -> None:
        if self.model_ema:
            self.model_ema.update(self.score)

    def compute_loss(self, x):
        

        self.score.train()
        
        with torch.no_grad():
            encodings = self.encode(x)


            if self.global_step ==0 or self.stat ==None:
              #  x_st = deconcat(x_0,modalities_list= self.modalities_list)
                self.stat_up = {}
                self.stat_up["cat"] = {}
                if self.stat == None:
                    for key in encodings.keys():
                        self.stat_up[key] = {}
                        self.stat_up[key]["std"] = encodings[key].std(dim = 0).detach()
                        self.stat_up[key]["mean"] = encodings[key].mean(dim = 0).detach()
                      #  self.stat_up[key]["min"] = encodings[key].min().detach()
                     #  self.stat_up[key]["max"] = encodings[key].max().detach()
                    print("before all preprocess")
                    print(self.stat_up)
                    self.stat = self.stat_up
                
                    with open(os.path.join(self.logger.log_dir,"stat.pickle"),"wb") as f:
                        pickle.dump(self.stat_up,f)

            if self.preprocess_type=="modality":
                encodings = self.preprocess(encodings)
                
            x_0 = concat_vect(encodings).clone().detach()
          
            
            if self.preprocess_type=="latent":
                x_0 = self.preprocess(x_0)
        
            if self.global_step ==0:
                x_st = deconcat(x_0,modalities_list= self.modalities_list)
                self.stat_up = {}
                self.stat_up["cat"] = {}
                self.stat_up["cat"]["std"] = x_0.std().detach()
                self.stat_up["cat"]["mean"] = x_0.mean().detach()
                self.stat_up["cat"]["min"] = x_0.min().detach()
                self.stat_up["cat"]["max"] = x_0.max().detach()
                #print(x_st)
                for key in x_st.keys():
                    self.stat_up[key] = {}
                    self.stat_up[key]["std"] = x_st[key].std(dim=0).detach()
                    self.stat_up[key]["mean"] = x_st[key].mean(dim=0).detach()
                    self.stat_up[key]["min"] = x_st[key].min().detach()
                    self.stat_up[key]["max"] = x_st[key].max().detach()
                print("after all preprocess")
                print(self.stat_up)
            
            
        loss = self.sde.train_step(x_0,self.score, nb_mods= len(self.modalities_list),d= self.d,
                                   subset_list_mask=self.mask_subset, modalities_list =self.modalities_list )
        
        loss_mod = deconcat(loss,self.modalities_list) 
        
        loss = 0.0
        for mod in self.modalities_list:
                loss += loss_mod[mod.name].sum() * mod.reconstruction_weight
        losses = {} 
  
    
        
        for mod in self.modalities_list:
            losses[mod.name] = loss_mod[mod.name].sum().detach() / self. batch_size 
        
        
        if  self.global_step %1000 == 0 and self.global_step   != 0 and self.debug:
                    self.do_sampling_and_cond_gen(step_log = self.global_step)
       
        return {"loss": loss / self.batch_size ,"unimodal_loss" : losses }
    
    def check_stat(self):
      
        self.stat = get_stat(data_loader= self.train_loader, 
                 modalities_list= list(self.modalities_list_dict.keys()),
                 device= self.device,
                 ae_model= self.aes)


    def encode(self, x):
        with torch.no_grad():
            return self.aes.encode(x)


    def decode(self, z):
        with torch.no_grad():
            return self.aes.decode(z)   

    def forward(self,x):
        with torch.no_grad():
            return self.aes.forward(x)
    
    
    def on_train_start(self):
        self.sde.device = self.device
      
        if self.do_check_stat  :
            self.check_stat()
    

    def sample(self, N):
        self.score.eval()
        if self.use_ema:
            self.model_ema.module.eval()
       # self.sde_reverse.device = self.device
        with torch.no_grad():
            x_c = torch.randn((N,self.latent_dim)).to(self.device)

            # x_c = self.sde.mod_repaint( score_net= self.score,x = x_c , subset= None, mask = torch.zeros_like(x_c))
            if self.use_ema:
                x_c  = self.sde.joint_gen(score_net = self.model_ema.module, x = x_c  )
            else:
                x_c  = self.sde.joint_gen(score_net = self.score, x = x_c  )

            if self.preprocess_type =="latent":
                x_c = self.postprocess(x_c)
            
            z_mods = deconcat(x_c, self.modalities_list)
            
            if self.preprocess_type =="modality":
                z_mods = self.postprocess(z_mods)
  
            decodings = self.decode(z_mods)
        return decodings


    def conditional_gen_latent_subsets(self, x, eps = 1e-5):
        return self.aes.conditional_gen_latent_subsets(x)
       

    def conditional_gen_all_subsets(self, x, N= None):
        self.score.eval()
        if self.use_ema:
            self.model_ema.module.eval()
        results = {}
        with torch.no_grad():
            

            encodings = self.encode(x)
            
            if self.preprocess_type=="modality":
                encodings = self.preprocess(encodings)
                
            x_0 = concat_vect(encodings).clone().detach()
            
            if self.preprocess_type=="latent":
                x_0 = self.preprocess(x_0)
         
            for idx, s_key in enumerate(self.subset_list_dict.keys()):
                if len(self.modalities_list) != len(self.subset_list_dict[s_key]):
                    mask = self.mask_subset[idx]["mask"].to(self.device)
                    mask = mask[:x_0.size(0)] 
                    x_subset = x_0 * mask + torch.randn_like(x_0) * (1.0 - mask) 
                    
                    if self.use_ema:
                    # x_c = self.sde.mod_repaint(score_net = self.score, x = x_subset , mask= mask ,subset=self.mask_subset[idx]["subset"])
                        x_c  = self.sde.mod_cross_gen(score_net = self.model_ema.module, x = x_subset , mask= mask ,subset=self.mask_subset[idx]["subset"])
                    else:
                        x_c  = self.sde.mod_cross_gen(score_net = self.score, x = x_subset , mask= mask ,subset=self.mask_subset[idx]["subset"])
                   
                else:
                    x_c = x_0

                
                if self.preprocess_type =="latent":
                     x_c = self.postprocess(x_c)

                z_mods = deconcat(x_c, self.modalities_list)
            
                if self.preprocess_type =="modality":
                     z_mods = self.postprocess(z_mods)
                    
                reconstruction = self.aes.decode(z_mods)

                results[s_key] = reconstruction
               
        return results 
    
    
    def conditional_gen_one_subsets(self, x, subset):
        self.score.eval()
        if self.use_ema:
            self.model_ema.module.eval()
        results = {}
        with torch.no_grad():
            

            encodings = self.aes.encode(x)
           
            
           
            if self.preprocess_type=="modality":
                encodings = self.preprocess(encodings)
                
            x_0 = concat_vect(encodings).clone().detach()
            
            if self.preprocess_type=="latent":
                x_0 = self.preprocess(x_0)
         
            for idx, s_key in enumerate(self.subset_list_dict.keys()):
                if s_key == subset:
           
                    mask = self.mask_subset[idx]["mask"].to(self.device)
                    mask = mask[:x_0.size(0)] 
                    x_subset = x_0 * mask + torch.randn_like(x_0) * (1.0 - mask) 
                    
                    if self.use_ema:
                    # x_c = self.sde.mod_repaint(score_net = self.score, x = x_subset , mask= mask ,subset=self.mask_subset[idx]["subset"])
                        x_c  = self.sde.mod_cross_gen(score_net = self.model_ema.module, x = x_subset , mask= mask ,subset=self.mask_subset[idx]["subset"])
                    else:
                        x_c  = self.sde.mod_cross_gen(score_net = self.score, x = x_subset , mask= mask ,subset=self.mask_subset[idx]["subset"])
                 
                    if self.preprocess_type =="latent":
                         x_c = self.postprocess(x_c)

                    z_mods = deconcat(x_c, self.modalities_list)
            
                    if self.preprocess_type =="modality":
                         z_mods = self.postprocess(z_mods)
                    
                    reconstruction = self.aes.decode(z_mods)

            return reconstruction 
    







    
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.score.parameters(), lr= self.lr)
        return optimizer
    
    
    
    
    
    def preprocess(self,encodings):
        if self.preprocess_op != None:
            if self.preprocess_op == "normalize":
                    encodings = self.normalize(encodings)
            elif self.preprocess_op == "standerdize":
                    encodings = self.standerdize(encodings)     
        return encodings
    
    
    def normalize(self, encodings):
        if self.preprocess_type =="latent":
            encodings = (encodings - self.stat["cat"]["min"] ) / (self.stat["cat"]["max"] - self.stat["cat"]["min"] ) 
        else:
            
            for key in encodings.keys():
                encodings[key] = (encodings[key] - self.stat[key]["min"] ) /(self.stat[key]["max"] - self.stat[key]["min"] ) 
        return encodings
    
    
    def denormalize(self, encodings):
        if self.preprocess_type =="latent":
            encodings = (encodings  * (self.stat["cat"]["max"] - self.stat["cat"]["min"] ) )  + self.stat["cat"]["min"]
        else:
            for key in encodings.keys():
                encodings[key] = (encodings[key]   * (self.stat[key]["max"] - self.stat[key]["min"] ) )  + self.stat[key]["min"]
        return encodings

 
    
    def postprocess(self,encodings):
        if self.preprocess_op == "normalize":
                encodings = self.denormalize(encodings)
        elif self.preprocess_op == "standerdize":
                encodings = self.destanderdize(encodings)
        return encodings
    
    
     
    def standerdize(self, encodings):
        if self.preprocess_type =="latent":
            encodings = (encodings - self.stat["cat"]["mean"] ) /(self.stat["cat"]["std"] ) 
        else:       
            for key in encodings.keys():
                encodings[key] = (encodings[key] - self.stat[key]["mean"] ) /(self.stat[key]["std"] ) 
               # encodings[key] = (encodings[key]  ) /(self.stat[key]["std"] ) 
        return encodings
     
     
    def destanderdize(self, encodings):
        if self.preprocess_type =="latent":
            encodings = (encodings  * (self.stat["cat"]["std"] ) )  + self.stat["cat"]["mean"]
        else:
            for key in encodings.keys():
                encodings[key] = (encodings[key]   * (self.stat[key]["std"] ) )  + self.stat[key]["mean"]
        return encodings