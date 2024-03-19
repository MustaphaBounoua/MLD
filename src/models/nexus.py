
import torch
import torch.nn as nn
from src.abstract.multimodal import MG
import numpy as np
from src.utils import stack_tensors
from src.unimodal.nexus_components.components import LinearEncoder, NexusEncoder 
from torch.distributions import Bernoulli
import torch.nn.functional as F

MODEL_STR = "nexus_impl"


BETAS = {
    "MHD":
    {
     "image":1.0,
     "sound":1.0,
     "trajectory":1.0,
     "label":1.0},
    "MNISTSVHN":
        {
     "mnist":1.0,
     "svhn":1.0,
        },
    "MMNIST":{
        "m0":1.0,
        "m1":1.0,
        "m2":1.0,
        "m3":1.0,
        "m4":1.0,
    },
       "CUB":{
           "image": 1.0, 
              "sentence":1.0}
    
}


GAMMAS = {
    "MHD":
    {
     "image":1.0,
     "sound":1.0,
     "trajectory":50.0,
     "label":50.0},
    "MNISTSVHN":
        {
     "mnist":3.92,
     "svhn":1.0,
        },
    "MMNIST":{
        "m0":1.0,
        "m1":1.0,
        "m2":1.0,
        "m3":1.0,
        "m4":1.0,
    },
    "CUB":{
    "image": 1.0, 
      "sentence":(64*64) / (32)}
    
}

CONFIG = { 
        "MNISTSVHN":{
        "nx":{
            "nexus_dim":20,
            "message_dim":256,
            "layer_sizes":[256, 256]
        },
        "mnist":{
            "mod_latent_dim": 16
        },
        "svhn":{
            "mod_latent_dim": 64
        }
        },
             
         "MHD":
        {
        "nx":{
            "nexus_dim":32,
            "message_dim":512,
            "layer_sizes":[512, 512]
        },
        "image":{
            "mod_latent_dim": 64
        },
        "sound":{
            "mod_latent_dim": 128
        },
        "trajectory":{
            "mod_latent_dim": 16
        },
        "label":{
            "mod_latent_dim":5
        },
    }
,
        "MMNIST":{
        "nx":{
            "nexus_dim":128,
            "message_dim":512,
            "layer_sizes":[512, 512]
        },
        "m0":{
            "mod_latent_dim": 160
        },
        "m1":{
            "mod_latent_dim": 160
        },
        "m2":{
            "mod_latent_dim": 160
        },
        "m3":{
            "mod_latent_dim": 160
        },
        "m4":{
            "mod_latent_dim": 160
        },
        },
    
       "CUB":{
        "nx":{
            "nexus_dim":64,
            "message_dim":512,
            "layer_sizes":[512, 512]
        },
        "image":{
            "mod_latent_dim": 64
        },
        "sentence":{
            "mod_latent_dim": 32
        }
        },
             
}



class Nexus_impl(MG):
    """ Product Expert
   Nexus Implementation

    """

    def __init__(self, latent_dim,
                 modalities_list, 
                 train_loader,
                 test_loader,
                 model_name=MODEL_STR, 
                 subsampling_strategy="powerset",
                 beta=1,
                 annealing_beta_gradualy=False,
                 nb_samples = 4, 
                 batch_size=256,
                 learning_paradigm ="unimodal_elbos",
                 num_train_lr = 500,
                 eval_epoch =5,
                 do_evaluation =True, 
                 do_fd = True,
                 log_epoch = 5,
                 n_fd = 5000,  limit_clip = 10000,
                 test_batch_size =256,
                 lr = 0.001,
                 dataset = "MHD",
                 max_epoch = 20,
  
                 ):
        self.learning_paradigm = learning_paradigm
        super(Nexus_impl, self).__init__(
                    latent_dim =latent_dim, 
                    modalities_list=   modalities_list, 
                    test_loader=test_loader,
                    train_loader=train_loader,
                    model_name=model_name,
                    subsampling_strategy=subsampling_strategy,
                    beta=beta,
                    batch_size=batch_size,
                    nb_samples = nb_samples, 
                    num_train_lr = num_train_lr ,
                    eval_epoch = eval_epoch,
                    do_evaluation=do_evaluation ,  
                    do_fd = do_fd,limit_clip =limit_clip,
                    log_epoch = log_epoch,
                    n_fd = n_fd, lr = lr ,do_class= False,
                    train_batch_size= test_batch_size
                    )
        self.max_epoch = max_epoch

        
        self.dataset = dataset
        self.config_nx= CONFIG[dataset]
        self.betas = BETAS[dataset]
        self.gammas = GAMMAS[dataset]
        self.nexus_encoders = self.set_nexus_encoders()
        self.nexus_decoders = self.set_nexus_decoders()
        self.nexus_node = self.set_nexus_node()
        self.aggregate_f = "mean_d"
        self.nx_drop_rate = 0.5
        self.training = True
        self.latent_dim = self.config_nx["nx"]["nexus_dim"]
        
    
    def get_beta(self, beta, epoch):
        if epoch>= self.max_epoch:
            return beta
        else:
            return  beta*(float(epoch+1)/self.max_epoch)
     
    def set_nexus_encoders(self):
        nx_encoders = nn.ModuleDict()
        for idx, modality in enumerate(self.modalities_list):
            nx_encoders[modality.name] = NexusEncoder(name=str(modality.name+'msg_encoder'),
                                          input_dim= self.config_nx[modality.name]['mod_latent_dim'],
                                          layer_sizes=[],
                                          output_dim=self.config_nx["nx"]['message_dim'])
        return nx_encoders

    def set_nexus_decoders(self):

        nx_decoder_layers = self.config_nx["nx"]['layer_sizes'] + [self.config_nx["nx"]['message_dim'] ]
        
        nx_decoders = nn.ModuleDict()
        for idx, modality in enumerate(self.modalities_list):
            nx_decoders[modality.name] = NexusEncoder(name=str('nexus_img_decoder'),
                                         input_dim=self.config_nx["nx"]['nexus_dim'],
                                         layer_sizes=nx_decoder_layers,
                                         output_dim=self.config_nx[modality.name]['mod_latent_dim'])
        return   nx_decoders

    def set_nexus_node(self):
        return LinearEncoder(name=str('nexus_updater'),
                                     input_dim=self.config_nx["nx"]['message_dim'],
                                     layer_sizes=self.config_nx["nx"]['layer_sizes'],
                                     output_dim=self.config_nx["nx"]["nexus_dim"])


    def encode(self, x):

        encodings = {}
        for idx, modality in enumerate(self.modalities_list):
            if modality.name in x.keys():
                mod_data = x[modality.name]
                mu_, logvar_ = self.encoders[idx](mod_data)
                encodings[modality.name] = [mu_, logvar_]

        return encodings
    
    

    def decode(self, z_mod):
        decodings = {}
        for idx, modality in enumerate(self.modalities_list):
            decodings[modality.name] = self.decoders[idx](z_mod[modality.name])
        return decodings


    def encode_nexus(self,z_mod):
        encodings = {}
        for key in z_mod.keys():
            encodings[key] = self.nexus_encoders[key](z_mod[key])
        return encodings



    def decode_nexus(self,z_nx):
        decodings = {}
        for idx, modality in enumerate(self.modalities_list):
            decodings[modality.name] = self.nexus_decoders[modality.name](z_nx)
        return decodings




    def forward(self, x):

        # Encode Modality Data
     
        encodings = self.encode(x)
        z_mods = {}
        z_mods_detached ={}
        for mod in encodings.keys():
            z_mods[mod] = self.reparam(*encodings[mod])
            z_mods_detached[mod] = z_mods[mod].clone().detach()
        reconstructions = self.decode(z_mods)


        encodings_nexus = self.encode_nexus(z_mods_detached)
        nx_msg = self.aggregate(encodings_nexus)

        nx_mu, nx_logvar = self.nexus_node(nx_msg)
        
        nx_z = self.reparam(nx_mu, nx_logvar)

        decodings_nexus = self.decode_nexus(nx_z)
        
        if self.dataset == "MHD":
            # print(encodings_nexus["sound"].shape)
            # print(decodings_nexus["sound"].shape)
            log_sigma = ((z_mods_detached["sound"] - decodings_nexus["sound"] ) ** 2).mean([0, 1], keepdim=True).sqrt().log()
        else:
            log_sigma = 0
        # reconstruction_nexus = self.decode(decodings_nexus)
        
        return {"unimodal_encodings" :encodings ,
                "unimodal_reconstructions" :reconstructions,
                "unimodal_z":z_mods_detached ,
                "encodings_nexus":encodings_nexus,
                "agg_vector" : nx_msg, 
                "nx_dist" : [nx_mu, nx_logvar],
                "nx_z": nx_z,
                "decodings_nx" : decodings_nexus ,
         #     "reconstruction_nexus":reconstruction_nexus  ,
                "logsigma" : log_sigma
        }


    def compute_loss_unimodals(self, results, x , batch_size):
        recons_log = {}
        kl_logs ={}

        reconstruction = results["unimodal_reconstructions"]
        distr_mod = results["unimodal_encodings"]

        logprobs = torch.zeros(len(x)).type_as(x[ list(x.keys())[0]])
        kl_unimodal = torch.zeros(len(x)).type_as(x[ list(x.keys())[0]])

        weights = torch.zeros(len(x)).type_as(logprobs)
        betas = torch.zeros(len(x)).type_as(logprobs)

        for  idx, key in enumerate( x.keys() ):
            mod = self.modalities_list_dict[key]
            if key == "sound" and mod.pretrained == True :
                pass
            else:
                logprobs[idx] = ( - mod.calc_log_prob( x[mod.name], reconstruction[mod.name] ) / batch_size )


                mod_logvar,mod_mu= distr_mod[key]
                kl_unimodal[idx] = (-0.5 * torch.sum(1 + mod_logvar - mod_mu.pow(2) - mod_logvar.exp())) / batch_size
          

                weights[idx] = float(mod.reconstruction_weight) 
                betas[idx] = self.get_beta(beta= float(self.betas[key] ), epoch= self.current_epoch )

                recons_log[mod.name] = logprobs[idx]
                kl_logs[key] = kl_unimodal[idx] 
                
                
        return  { "loss": (weights * logprobs + betas * kl_unimodal  ).sum(dim=0), "rec_loss": recons_log, "kl_logs" :kl_logs }


    def gaussian_nll(self, mu, log_sigma, x):
  
        return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)


    def compute_loss_nexus(self, results, batch_size):
        loss_recon = 0
        loss_recon_log={}
        for key in results["unimodal_z"].keys():
            ## treat sound here
            if key =="sound" :
                nx_snd_log_sigma = ((results["unimodal_z"][key].clone().detach() - results["decodings_nx"][key]) ** 2).mean([0, 1], keepdim=True).sqrt().log()

                loss_mod = self.gammas[key] * torch.sum(self.gaussian_nll(results["decodings_nx"][key], nx_snd_log_sigma , results["unimodal_z"][key].clone().detach()))
                
            else:
                loss_mod = self.gammas[key] * torch.sum( F.mse_loss(input = results["unimodal_z"][key].detach(),
                                                        target = results["decodings_nx"][key],
                                                        reduction='none'))
            loss_recon_log[key] = torch.sum(loss_mod)/batch_size
            loss_recon += loss_mod


        nx_mu,nx_logvar =results["nx_dist"]
        # Nexus
        beta= self.get_beta(beta= float(self.beta ), epoch= self.current_epoch )
        nx_prior = beta * (-0.5 * torch.sum(1 + nx_logvar - nx_mu.pow(2) - nx_logvar.exp()))
        # Total loss
        loss = (loss_recon + nx_prior) / batch_size

        return {"loss": loss , "kld": torch.sum(nx_prior) / batch_size, "unimodal_loss":loss_recon_log }


  


    def compute_loss(self, x):
        self.train()
        """_summary_
            compute the elbo loss as defined in the paper :
            Elbo(S1)+...+Elbo(Sn) where Sn are the subsets see --> utils.subset
        Returns:
            loss: Elbo loss
        """
        # get the encoding of all modalities present in x

        # training x should be without missing modality.
        
     
        self.is_training = True
        
        results = self.forward(x)
        loss_unimodals = self.compute_loss_unimodals(results, x , self.batch_size)
        loss_nexus = self.compute_loss_nexus(results,self.batch_size)

        total_loss = loss_unimodals["loss"] + loss_nexus["loss"]
           
       
        
        
        return {"loss" : total_loss, 
                "KLD_joint" :loss_nexus["kld"]  , 
                "KLDs": loss_unimodals["kl_logs"]  , 
                "Rec_loss":loss_unimodals["rec_loss"],
                "loss_nx":loss_nexus["loss"],
                "loss_mod":loss_unimodals["loss"],
                "Rec_nex": loss_nexus["unimodal_loss"]
                 }

        

    
    def elbo_objectif(self, reconstruction_error, KLD, beta):

        return (reconstruction_error + beta * KLD) 
    

    def compute_KLD(self,posterior,batch_size):
        """
            compute the reconstrucntion loss for a single forward pass
        """
        mu_joint, logvar_joint = posterior["joint"]
        return self.Kl_div_gaussian(mu_joint,logvar_joint)/batch_size

    def conditional_gen_all_subsets(self, x, N =8):
        self.eval()
        self.is_training = False
        results = {}
        modalities_str = np.array([mod.name for mod in self.modalities_list])
        subsets = { ','.join(modalities_str[s]) : s for s in self.subset_list}
            
        with torch.no_grad():
            encodings = self.encode(x)
            for idx, s_key in enumerate(subsets):
                sub_encodings = {
                    modalities_str[mod_i] : encodings[modalities_str[mod_i]]   for mod_i in subsets[s_key]
                }
                z_mods_sub={}
                for mod in sub_encodings.keys():
                    z_mods_sub[mod] = self.reparam(*sub_encodings[mod])
                
                encodings_nexus = self.encode_nexus(z_mods_sub)
                
                nx_msg = self.aggregate(encodings_nexus)
              
                nx_mu, nx_logvar = self.nexus_node(nx_msg)
                nx_z = self.reparam(nx_mu, nx_logvar)
                decodings_nexus = self.decode_nexus(nx_z)
                reconstruction_nexus = self.decode(decodings_nexus)
                # ## 
                # z_mods_sub={} 
                # for mod in modalities_str:
                #     z_mods_sub[mod] = self.reparam(*encodings[mod])
                
               # reconstruction_nexus = self.decode(z_mods_sub)


                results[s_key] = reconstruction_nexus    
        return results 
    
    
    
    def conditional_gen_latent_subsets(self, x):
        self.eval()
        self.is_training = False
        results = {}
        modalities_str = np.array([mod.name for mod in self.modalities_list])
        subsets = { ','.join(modalities_str[s]) : s for s in self.subset_list}
            
        with torch.no_grad():
            encodings = self.encode(x)
            for idx, s_key in enumerate(subsets):
                sub_encodings = {
                    modalities_str[mod_i] : encodings[modalities_str[mod_i]]   for mod_i in subsets[s_key]
                }
                z_mods_sub={}
                for mod in sub_encodings.keys():
                    z_mods_sub[mod] = self.reparam(*sub_encodings[mod])
                
                encodings_nexus = self.encode_nexus(z_mods_sub)
                
                nx_msg = self.aggregate(encodings_nexus)
              
                nx_mu, nx_logvar = self.nexus_node(nx_msg)
                
                results[s_key] = [nx_mu, nx_logvar]
               
        return results 
    
    
    
    def sample(self, N):
        self.eval()
        with torch.no_grad():
            nx_z = torch.randn(N, self.latent_dim, device=self.device)
            decodings_nexus = self.decode_nexus(nx_z)
            reconstruction_nexus = self.decode(decodings_nexus)
            return reconstruction_nexus
    
    def compute_reconstruction_error(self):
        pass
    
    
    
    
    def mean_drop(self, mean_msg, mod_msgs):

        # Compute mean message
        mean_msg = torch.mean(mean_msg, dim=0)

        if self.is_training == False:
            return mean_msg
        else:
            # For each entry in batch: (During training we have all modalities available)
            for i in range(mean_msg.size(0)):

                drop_mask = Bernoulli(torch.tensor([self.nx_drop_rate])).sample()

                # If there is no d, we continue
                if torch.sum(drop_mask).item() == 0:
                    continue

                # If there is d, we randomly select the number and type of modalities to drop
                else:
                    n_mods_to_drop = torch.randint(low=1, high=len(mod_msgs), size=(1,)).item()
                    mods_to_drop = np.random.choice(range(len(mod_msgs)), size=n_mods_to_drop, replace=False)

                    prune_msg = torch.zeros(mod_msgs[0].size(-1)).to(self.device)
                    prune_msg = prune_msg.unsqueeze(0)
                    # if self.use_cuda:
                    #     prune_msg = prune_msg.cuda()

                    for j in range(len(mod_msgs)):
                        if j in mods_to_drop:
                            continue
                        else:
                            prune_msg = torch.cat([prune_msg, mod_msgs[j][i].unsqueeze(0)], dim=0)
                    prune_msg = prune_msg[1:]
                    mean_msg[i] = torch.mean(prune_msg, dim=0)

            return mean_msg
        
        
    
    def aggregate(self, encodings_nx):
        
        comp_msg = stack_tensors(encodings_nx)
        
        if len(encodings_nx)==1:
            return comp_msg[0]
    
        #comp_msg = stack_tensors(encodings_nx)

        # Aggregate
        if self.aggregate_f == 'mean':
            comp_msg = torch.mean(comp_msg, dim=0)
        elif self.aggregate_f == 'mean_d':
            comp_msg = self.mean_drop(comp_msg, [value for key,value in encodings_nx.items() ])
        elif self.aggregate_f == 'sum':
            comp_msg = torch.sum(comp_msg, dim=0)
        else:
            raise ValueError("Not implemented")
        return comp_msg
