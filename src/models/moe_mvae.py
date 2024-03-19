
import numpy as np
import torch
import torch.nn as nn
from src.abstract.multimodal import MG
from src.utils import stack_posterior

MODEL_STR = "MoE_MVAE"


class MoEVAE(MG):
    """ Mixture of Expert
    Implementation of the mmvae model 

    https://arxiv.org/abs/1911.03393

    The model extends the Base_MVAE abstract class.

    """

    def __init__(self, latent_dim,
                 modalities_list, 
                 train_loader,
                 test_loader,
                 model_name=MODEL_STR, 
                 subsampling_strategy="powerset",
                 beta=1,
                 annealing_beta_gradualy=False,
                 nb_samples = 8, 
                 batch_size=256,
                 num_train_lr = 500,
                 eval_epoch= 5,
                 do_evaluation=True,
                 do_fd = True,
                 log_epoch = 5,
                 n_fd = 5000,
                 limit_clip = 3000,
                 lr = 0.001,
                 test_batch_size = 256,
                 dataset=None
                 ):
        super(MoEVAE, self).__init__(
                latent_dim =latent_dim, 
                modalities_list=   modalities_list, 
                test_loader=test_loader,
                train_loader=train_loader,
                model_name=model_name,
                subsampling_strategy=subsampling_strategy,
                beta=beta,
                batch_size=batch_size,
                nb_samples = nb_samples, 
                num_train_lr = num_train_lr,
                eval_epoch = eval_epoch,do_evaluation=do_evaluation,
                do_fd = do_fd,limit_clip =limit_clip,
                do_class= False,
                log_epoch = log_epoch,
                n_fd = n_fd,
                lr = lr,
                train_batch_size=test_batch_size,dataset=dataset
                )

        self.posterior = MixtureOfExperts()
     
       

    def compute_loss(self, x):
        
        """_summary_
            compute the elbo loss as defined in the paper.        
        Returns:
            loss: Elbo loss
        """
        # get the encoding of all modalities present in x

        # training x should be without missing modality.
        encodings = self.encode(x)
        loss = 0
        posterior = self.posterior(encodings)
        
        mu_joint, logvar_joint = posterior["joint"]
        z = self.reparam(mu_joint, logvar_joint)
        reconstruction = self.decode(z)
        
        reconstruction_error = self.compute_reconstruction_error(x, reconstruction , self.batch_size)
        KLD = self.compute_KLD(posterior, self.batch_size)
        
        loss = self.elbo_objectif(reconstruction_error= reconstruction_error["weighted"],KLD= KLD["KLD_joint"], beta=self.beta)
        
        #print({"loss" : loss, "KLD_joint" :KLD["KLD_joint"], "KLDs":KLD["KLDs"], "Rec_loss": reconstruction_error["rec_loss"] })
        #print({"recon_loss":reconstruction_error["weighted"],"beta":self.beta})
         
        return {"loss" : loss, "KLD_joint" :KLD["KLD_joint"], "KLDs":KLD["KLDs"], "Rec_loss": reconstruction_error["rec_loss"] }


    
    def elbo_objectif(self, reconstruction_error, KLD, beta):
        return reconstruction_error + beta * KLD
    

    def compute_KLD(self, posterior,batch_size):

        encodings = posterior["individual"]
        
        mu = posterior["joint"][0]
        
        num_mod= len(encodings)
        
        weights = (1/float(num_mod))*torch.ones(num_mod).type_as(mu)
        klds = torch.zeros(num_mod).type_as(mu)
        
        kl_joint = 0
        kld_mods ={}
        for idx, key in enumerate(encodings.keys()) :
            mu,logvar = encodings[key]
            
            kl_mod=  self.Kl_div_gaussian(mu,logvar) / batch_size 
            kld_mods[key]= kl_mod
            kl_joint += kl_mod
            klds[idx] = kl_mod
        
        return {"KLD_joint"  :  (weights*klds).sum(dim=0) , "KLDs": kld_mods   }
    


    def compute_reconstruction_error(self, x, reconstruction, batch_size):
        recons_log = {}
        
        logprobs = torch.zeros(len(x)).type_as(x[self.modalities_list[0].name])
        weights = torch.zeros(len(x)).type_as(x[self.modalities_list[0].name])
        for  idx, mod in enumerate( self.modalities_list ):
            logprobs[idx] = ( - mod.calc_log_prob( x[mod.name], reconstruction[mod.name] ) / batch_size )
            weights[idx] = float(mod.reconstruction_weight)  
            recons_log[mod.name] = logprobs[idx]
        return  { "weighted": (weights*logprobs).sum(dim=0), "rec_loss": recons_log}
    



   
    def conditional_gen_all_subsets(self, x , N =None):
        
        results = {}
        modalities_str = np.array([mod.name for mod in self.modalities_list])
        subsets = { ','.join(modalities_str[s]) : s for s in self.subset_list}
            
        with torch.no_grad():
            encodings = self.encode(x)
            for idx, s_key in enumerate(subsets):
                sub_encodings = {
                    modalities_str[mod_i] : encodings[modalities_str[mod_i]]   for mod_i in subsets[s_key]
                }
                mu,logvar = self.posterior(sub_encodings) ["joint"]
                z = self.reparam(mu,logvar)
                results[s_key] = self.decode(z)
               
        return results 
    
    
    
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
                mu,logvar = self.posterior(sub_encodings) ["joint"]
                
                results[s_key] = [mu,logvar]
               
        return results 
    


class MixtureOfExperts(nn.Module):
    """Return parameters for product of independent experts as implemented in:
    See https://github.com/thomassutter/MoPoE

   
    
    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """

    def forward(self, encodings , weights= None):
         
        mus, logvars = stack_posterior(encodings)
        mu_joint,logvar_joint = mixture_component_selection(mus, logvars)
        
        # batch_size = mus.size(1)
        # nb_modalties = len(encodings)
        # weight = 1/nb_modalties
        # ## x.shape = (nb_modalities, batch_size, input size)
        # step = int( batch_size * weight )
        # indices = np.arange(start= 0, stop = batch_size+1, step = step )
        # indices[-1] = batch_size
        # modalities_order= np.arange(0,nb_modalties,1)
        # #np.random.shuffle(modalities_order)
        # mu_joint = torch.cat([mus[k, indices[idx]:indices[idx+1], :] for   idx, k in enumerate (modalities_order)])
        # logvar_joint = torch.cat([logvars[k, indices[idx]:indices[idx+1], :] for   idx, k in enumerate (modalities_order)])

        return {"joint": [mu_joint, logvar_joint],"individual":encodings }
    
    
    
def mixture_component_selection(mus, logvars):
       
        num_components = mus.shape[0];
        num_samples = mus.shape[1];
        
        w_modalities = (1/float(num_components))*torch.ones(num_components)
        
        idx_start = [];
        idx_end = []
        for k in range(0, num_components):
            if k == 0:
                i_start = 0;
            else:
                i_start = int(idx_end[k-1]);
            if k == w_modalities.shape[0]-1:
                i_end = num_samples;
            else:
                i_end = i_start + int(torch.floor(num_samples * w_modalities[k]));
            idx_start.append(i_start);
            idx_end.append(i_end);
        idx_end[-1] = num_samples;
        mu_sel = torch.cat([mus[k, idx_start[k]:idx_end[k], :] for k in range(w_modalities.shape[0])]);
        logvar_sel = torch.cat([logvars[k, idx_start[k]:idx_end[k], :] for k in range(w_modalities.shape[0])]);
        return mu_sel, logvar_sel
    
    
 