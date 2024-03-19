
import torch
import torch.nn as nn
from src.abstract.multimodal import MG
import numpy as np
from src.utils import stack_posterior


MODEL_STR = "MVTCAE"


class MVTCAE(MG):
    """ Total correlation

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
                 nb_batchs = 10,
                 n_fd = 5000,
                 tc_ratio = 5/6,
                 lr = 0.001,
                 test_batch_size = 256,dataset =None , limit_clip = 5000,
                 ):
        self.learning_paradigm = learning_paradigm
        self.tc_ratio = tc_ratio
        super(MVTCAE, self).__init__(
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
                    do_fd = do_fd,
                    limit_clip = limit_clip,
                    nb_batchs= nb_batchs,
                    log_epoch = log_epoch,do_class= False,
                    n_fd = n_fd, lr = lr , train_batch_size= test_batch_size,dataset=dataset )
        
        
        self.posterior = ProductOfExperts()
      

    def compute_loss(self, x):
        """_summary_
            compute the elbo loss as defined in the paper :
            Elbo(S1)+...+Elbo(Sn) where Sn are the subsets see --> utils.subset
        Returns:
            loss: Elbo loss
        """
        # get the encoding of all modalities present in x

        # training x should be without missing modality.
        encodings = self.encode(x)
       
        ## full subset
        posterior = self.posterior(encodings)
        mu_joint, logvar_joint = posterior["joint"]
        z = self.reparam(mu_joint, logvar_joint)
        reconstruction = self.decode(z)
        reconstruction_error_joint = self.compute_reconstruction_error(x, reconstruction,batch_size = self.batch_size)
        
        kld_joint = self.Kl_div_gaussian(mu_joint,logvar_joint) /  self.batch_size
        kld_vib,kld_vib_log = self.compute_KLDs_vib(posterior,self.batch_size)

        elbo_loss = self.elbo_objectif(reconstruction_error= reconstruction_error_joint["weighted"],
                                        KLD=kld_joint,KLD_VIB= kld_vib,n_mod= len(encodings),tc_ratio= self.tc_ratio,
                                         beta=self.beta 
                                         )
        return {"loss" : elbo_loss, "KLD_joint" :kld_joint, "KLDs":kld_vib_log, 
                "Rec_loss": reconstruction_error_joint["rec_loss"]  }

        

    
    def elbo_objectif(self, reconstruction_error, KLD, KLD_VIB,n_mod,tc_ratio, beta):

        rec_weight = (n_mod - tc_ratio) / n_mod
        cvib_weight = tc_ratio / n_mod  # 0.3
        vib_weight = 1 - tc_ratio  # 0.1
        kld_weighted = cvib_weight * KLD_VIB + vib_weight * KLD
        total_loss = rec_weight * reconstruction_error + beta * kld_weighted
        return total_loss
    

    def compute_KLD(self,posterior,batch_size):
        """
            compute the reconstrucntion loss for a single forward pass
        """
        mu_joint, logvar_joint = posterior["joint"]
        return self.Kl_div_gaussian(mu_joint,logvar_joint)/batch_size


    

    def compute_KLDs_vib(self, posterior,batch_size):

        encodings = posterior["individual"]
        
        mu_joint,logvar_joint = posterior["joint"]
        
        num_mod= len(encodings)
        
        #weights = (1/float(num_mod))*torch.ones(num_mod).type_as(mu)
        klds = torch.zeros(num_mod).type_as(mu_joint)
        
        kld_mods ={}
        for idx, key in enumerate(encodings.keys()) :
            mu,logvar = encodings[key]
            
            kl_mod=  self.vib_Kl_div_gaussian(mu_joint,logvar_joint,mu,logvar) / batch_size 

            kld_mods[key]= kl_mod
            klds[idx] = kl_mod
        
        return klds.sum(dim=0) ,  kld_mods   

    

    def vib_Kl_div_gaussian(self,mu0,logvar0,mu1,logvar1):
        KLD = -0.5 * (torch.sum(1 - logvar0.exp()/logvar1.exp() - (mu0-mu1).pow(2)/logvar1.exp() + logvar0 - logvar1))
        return KLD




    
    def compute_reconstruction_error(self, x, reconstruction, batch_size):
        recons_log = {}
        
        logprobs = torch.zeros(len(x)).type_as(x[ list(x.keys())[0]])
        weights = torch.zeros(len(x)).type_as(logprobs)
        
        for  idx, key in enumerate( x.keys() ):
            if key in self.modalities_list_dict.keys():
                mod = self.modalities_list_dict[key]
                logprobs[idx] = ( - mod.calc_log_prob( x[mod.name], reconstruction[mod.name] ) / batch_size )
                if len(x) == len(self.modalities_list):
                    weights[idx] = float(mod.reconstruction_weight)  
                else : 
                    weights[idx] = 1.0
                recons_log[mod.name] = logprobs[idx]
        return  { "weighted": (weights*logprobs).sum(dim=0), "rec_loss": recons_log}
    


    
    
    

    def conditional_gen_all_subsets(self, x, N =None):
        
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
        
        return {"joint": [pd_mu, pd_logvar], "individual":encodings }






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
