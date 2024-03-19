
import torch
import torch.nn as nn

from src.abstract.multimodal import MG
import numpy as np
from src.utils import stack_posterior


MODEL_STR = "PoE_MVAE"


class PoEVAE(MG):
    """ Product Expert
    Implementation of the mvae model 

    https://arxiv.org/abs/1802.05335

    The model extends the GenerativeModel abstract class in order to be able to work with the eval methods.

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
                 learning_paradigm ="unimodal_elbos",
                 num_train_lr = 500,
                 eval_epoch =5,
                 do_evaluation =True, 
                 do_fd = True,
                 log_epoch = 5,
                 n_fd = 5000, 
                 lr = 0.001,
                 test_batch_size =256,
                 dataset= None,
                 limit_clip = 3000,nb_batchs = None
                 ):
        self.learning_paradigm = learning_paradigm
        super(PoEVAE, self).__init__(
                    latent_dim =latent_dim, 
                    modalities_list=   modalities_list, 
                    test_loader=test_loader,
                    train_loader=train_loader,
                    model_name=model_name,
                    subsampling_strategy=subsampling_strategy,
                    beta=beta,nb_batchs= nb_batchs,
                    batch_size=batch_size,
                    nb_samples = nb_samples, 
                    num_train_lr = num_train_lr ,
                    eval_epoch = eval_epoch,
                    do_evaluation=do_evaluation ,  
                    do_fd = do_fd,
                    log_epoch = log_epoch,
                    n_fd = n_fd, lr = lr,do_class= False,
                    train_batch_size= test_batch_size, limit_clip= limit_clip,
                    dataset=dataset)
        
        self.posterior = ProductOfExperts()
      

    def compute_loss(self, x):
        """_summary_
            compute the elbo loss as defined in the paper :
            Elbo(S1)+...+Elbo(Sn) where Sn are the subsets see --> utils.subset
        Returns:
            loss: Elbo loss
        """
        
        # get the encoding of all modalities present in x
        self.train()
        # training x should be without missing modality.
        encodings = self.encode(x)
       
        ## full subset
        posterior = self.posterior(encodings)
        mu_joint, logvar_joint = posterior["joint"]
        z = self.reparam(mu_joint, logvar_joint)
        reconstruction = self.decode(z)
        reconstruction_error_joint = self.compute_reconstruction_error(x, reconstruction,batch_size = self.batch_size)
        
        kld_joint = self.Kl_div_gaussian(mu_joint,logvar_joint) /  self.batch_size
        loss_joint = self.elbo_objectif(reconstruction_error_joint["weighted"], kld_joint, beta=self.beta)
        
        elbo_loss={}
        klds = {}
        loss_unimodal = 0
        
        klds["joint"] = kld_joint
        elbo_loss["joint"] = loss_joint
        
        if ( self.learning_paradigm =="unimodal_elbos" ):
            
            for idx, modality in enumerate(self.modalities_list):
                posterior = self.posterior( {modality.name :  encodings[modality.name]} )
                
                mu_joint, logvar_joint = posterior["joint"]
                z = self.reparam(mu_joint, logvar_joint)
                
                reconstruction = self.decode(z)
                reconstruction_error = self.compute_reconstruction_error({modality.name:x[modality.name] }, 
                                                                         {modality.name:reconstruction[modality.name]},batch_size = self.batch_size )
                
                kld = self.compute_KLD(posterior,self.batch_size)
                klds[modality.name] = kld
                loss_modality = self.elbo_objectif(reconstruction_error["weighted"],kld, beta=self.beta)
                elbo_loss[modality.name] = loss_modality
                loss_unimodal = loss_unimodal + loss_modality
                
        total_loss = loss_joint + loss_unimodal 
        elbo_loss["total"] = total_loss
           
        
        return {"loss" : total_loss, "KLD_joint" :klds["joint"], "KLDs":klds, 
                "Rec_loss": reconstruction_error_joint["rec_loss"] , "unimodal_elbos":elbo_loss }

        

    
    def elbo_objectif(self, reconstruction_error, KLD, beta):

        return (reconstruction_error + beta * KLD) 
    

    def compute_KLD(self,posterior,batch_size):
        """
            compute the reconstrucntion loss for a single forward pass
        """
        mu_joint, logvar_joint = posterior["joint"]
        return self.Kl_div_gaussian(mu_joint,logvar_joint)/batch_size

    
    
    
    def compute_reconstruction_error(self, x, reconstruction, batch_size):
        recons_log = {}
        
        logprobs = torch.zeros(len(x)).type_as(x[ list(x.keys())[0]])
        weights = torch.zeros(len(x)).type_as(logprobs)
        
        for  idx, key in enumerate( x.keys() ):
            mod = self.modalities_list_dict[key]
            logprobs[idx] = ( - mod.calc_log_prob( x[mod.name], reconstruction[mod.name] ) / batch_size )
            if len(x) == len(self.modalities_list):
                weights[idx] = float(mod.reconstruction_weight)  
            else : 
                weights[idx] = 1.0
            recons_log[mod.name] = logprobs[idx]
        return  { "weighted": (weights*logprobs).sum(dim=0), "rec_loss": recons_log}
    


    
    
    

    def conditional_gen_all_subsets(self, x,N =None):
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
                
                mu,logvar = self.posterior(sub_encodings) ["joint"]
                z = self.reparam(mu,logvar)
                results[s_key] = self.decode(z)
                
        return results 
    
    
    
    def conditional_gen_latent_subsets(self, x):
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
                mu,logvar = self.posterior(sub_encodings) ["joint"]
                
                results[s_key] = [mu,logvar]
               
        return results 
    
    

class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """

    def forward(self,  encodings, eps=1e-8 , add_prior = True):
        mu,logvar = stack_posterior( encodings)
        
        if add_prior:
            
            mu_prior = torch.zeros(1, mu.size()[1], mu.size()[2]).type_as(mu)
            logvar_prior = torch.zeros(1, mu.size()[1], mu.size()[2]).type_as(mu)

            mu = torch.cat((mu, mu_prior), dim=0)
            logvar = torch.cat((logvar,logvar_prior), dim=0)

        
        var = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / (var)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0) 
        pd_var = 1. / torch.sum(T, dim=0) 
        pd_logvar = torch.log(pd_var + eps)
        
        return {"joint": [pd_mu, pd_logvar], "individual":[mu,logvar] }






def prior_expert(size):
    """Universal prior expert. Here we use a spherical
    Gaussian: N(0, 1).

    @param size: integer
                dimensionality of Gaussian
    @param use_cuda: boolean [default: False]
                    cast CUDA on variables
    """
    mu = torch.zeros(size)
    logvar = torch.zeros(size)
    return mu, logvar
