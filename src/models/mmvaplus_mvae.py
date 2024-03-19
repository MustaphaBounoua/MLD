
import numpy as np
import torch
import torch.nn as nn
from src.abstract.multimodal import MG
from src.utils import stack_posterior,cat_shared_private, log_mean_exp
import torch.distributions as dist
import torch.nn.functional as F
MODEL_STR = "MMVAE plus"


class MMVAE_plus(MG):
    """ Mixture of Expert
    Implementation of the mmvae model 

    https://arxiv.org/abs/1911.03393

    The model extends the Base_MVAE abstract class.

    """

    def __init__(self, 
                 latent_dim,
                 latent_dim_w,
                 modalities_list, 
                 train_loader,
                 test_loader,
                 model_name=MODEL_STR, 
                 subsampling_strategy="unimodal",
                 beta=1,
                 annealing_beta_gradualy=False,
                 nb_samples = 8, 
                 batch_size=256,
                 K=1,
                 elbo ="iwae",
                 num_train_lr = 500,
                 eval_epoch= 5,
                 do_evaluation=True,
                 do_fd = True,
                 log_epoch = 5,
                 n_fd = 5000,
                 limit_clip = 3000,
                 lr = 0.001,
                 nb_batchs = 10,
                 test_batch_size = 256,
                 dataset=None,
                 ):
        super(MMVAE_plus, self).__init__(
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
                eval_epoch = eval_epoch,
                do_evaluation=do_evaluation,
                do_fd = do_fd,limit_clip =limit_clip,
                do_class= False,
                log_epoch = log_epoch,
                n_fd = n_fd,
                lr = lr,
                nb_batchs=nb_batchs,
                train_batch_size=test_batch_size,dataset=dataset
                )

        self.posterior = MixtureOfExpertsPlus()
        self.latent_dim_w = latent_dim_w
        grad_w = {'requires_grad': True}
        self.K =K
        self.elbo = elbo
        self.pw = dist.Laplace
        self.pw_params = nn.ParameterDict({mod.name :
            nn.ParameterList([
            nn.Parameter(torch.zeros(1, latent_dim_w), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, latent_dim_w), **grad_w)  # logvar
        ]) for mod in modalities_list
        })

        
        
        self.pu = dist.Laplace
        self._pu_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, latent_dim + latent_dim_w), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, latent_dim + latent_dim_w),requires_grad= False)  # logvar
        ])
    
    @property
    def pu_params(self):
        return self._pu_params[0], F.softmax(self._pu_params[1], dim=1) * self._pu_params[1].size(-1)


    def encode(self, x):

        encodings = {}
        for idx, modality in enumerate(self.modalities_list):
            if modality.name in x.keys():
                mod_data = x[modality.name]
                mu_w, logvar_w ,mu_u, logvar_u  = self.encoders[idx](mod_data)
                encodings[modality.name]={
                    "shared":[mu_u, logvar_u],
                    "private":[mu_w, logvar_w]
                }
        return encodings

    def decode_train(self, posteriors_z,flatten=False):
        output = {key:{} for key in posteriors_z.keys()}
        for mod_in in posteriors_z.keys() :
            for idx, mod in enumerate(self.modalities_list):
                mod_out = mod.name
                z = posteriors_z[mod_in][mod_out]
                rec = self.decoders[idx](z)
                if flatten:
                    output[mod_in][mod_out] =  rec.view(-1,*mod.size).detach()
                else:
                    output[mod_in][mod_out] =  rec
        return output
    
    def decode(self, z):
        output = {}
        for idx, mod in enumerate(self.modalities_list):
                mod_out = mod.name
              
                out = self.decoders[idx](z)
              
                output[mod_out] = out.view(-1,*mod.size)
        return output

    def sample(self, N):
        self.eval()
        with torch.no_grad():
            z = self.pu(*self.pu_params).rsample([N])
            return self.decode(z)


    def compute_loss(self, x):
        # self.do_sampling_and_cond_gen()
        #self.final_eval()

        encodings = self.encode(x)
        
        posteriors, u_s = self.posterior(encodings, self.pw ,self.pu,self.pw_params,self.latent_dim,self.latent_dim_w,K=self.K )
        pxu = self.decode_train(u_s)

       # loss, kld, logpx = self.m_elbo_iwae(posteriors,u_s,pxu,x)
        if self.elbo == "iwae":
            loss , logs = self.m_elbo_iwae(posteriors,u_s,pxu,x)
        elif self.elbo== "dreg":
            loss , logs = self.m_elbo_dreg(posteriors,u_s,pxu,x)

        logs_recon,logs_kld_pu ,logs_kld_z ,logs_kld_w =  logs

        return {"loss" : -loss,"logs_recon":logs_recon,"logs_kld_pu":logs_kld_pu,"logs_kld_z":logs_kld_z,"logs_kld_w":logs_kld_w}


    def m_elbo_iwae(self,posterior,u_s, pxu,x):
 
        qz_xs={}
        qw_xs={}
        lws = []
        for mod_in in posterior.keys():
            w_param = posterior[mod_in][mod_in]["private"]
            z_param= posterior[mod_in][mod_in]["shared"]
            qw_xs[mod_in] = self.pu(*w_param)
            qz_xs[mod_in] = self.pu(*z_param)

        logs_recon={}
        logs_kld_pu = {}
        logs_kld_z = {}
        logs_kld_w = {}

        for mod_in in posterior.keys():

            u = u_s[mod_in][mod_in]

            lpu = self.pu(*self.pu_params).log_prob(u).sum(-1)
            #lpu_klds.append(lpu)
            # print("lpu.shape")
            # print(lpu.shape)
            # print(lpu[0][0])
          #  print(lqw_x.shape)
         

            logs_kld_pu[mod_in] = lpu.clone().detach().sum()/u.size(1)

            w, z = torch.split(u, [self.latent_dim_w, self.latent_dim], dim=-1)

            lqz_x = log_mean_exp(torch.stack([qz_xs[qz_x_key].log_prob(z).sum(-1) for qz_x_key in qz_xs.keys()]))

            # print("lqz_x.shape")
            # print(lqz_x.shape)
            # print(lqz_x[0][0])
        
            logs_kld_z[mod_in] = lqz_x.clone().detach().sum()/u.size(1)

            lqw_x = qw_xs[mod_in].log_prob(w).sum(-1)
            # print("lqw_x.shape")
            # print(lqw_x.shape)
            # print(lqw_x[0][0])

            logs_kld_w[mod_in] = lqw_x.clone().detach().sum()/u.size(1)

          #  print("lqw_x.shape")
          #  print(lqw_x.shape)

            lpx_u =  self.compute_reconstruction_error( x, pxu[mod_in])
            # print("lpx_u.shape")
            # print(lpx_u.shape)
            # print(lpx_u[0][0])
         
            #print("the loss mean exp"+ str(lpx_u)) 
           
            # print("lpx_u.shape")
            # print(lpx_u.shape)

            logs_recon[mod_in] = lpx_u.clone().detach().sum()/u.size(1)
            #lpx_us.append(lpx_u)
            

            lw = lpx_u + self.beta*(lpu - lqz_x - lqw_x)

           

            lws.append(lw)

        lws= torch.stack(lws) 
        # print("lws.shape")
        # print(lws.shape)
        # print(lpx_u[0][0])
       
        loss = log_mean_exp(lws, dim=1).mean(0)
        # print("loss.shape")
        # print(loss.shape)
        # print(loss[0])
        # print("loss.sum")
        # print(loss.sum())
        # print("lws.shape")
        # print(lws.shape)
        return loss.mean(), [  logs_recon,logs_kld_pu ,logs_kld_z ,logs_kld_w ]
    



    def m_elbo_dreg(self,posterior,u_s, pxu,x):
 
        qz_xs={}
        qw_xs={}
        lws = []

         
      
        for mod_in in posterior.keys():
            w_param = posterior[mod_in][mod_in]["private"]
            w_param = [ w_param[0].detach(), w_param[1].detach()]
            z_param = posterior[mod_in][mod_in]["shared"]
            z_param = [ z_param[0].detach(), z_param[1].detach()]

            qw_xs[mod_in] = self.pu(*w_param)
            qz_xs[mod_in] = self.pu(*z_param)

        logs_recon={}
        logs_kld_pu = {}
        logs_kld_z = {}
        logs_kld_w = {}
        uss = []
        for mod_in in posterior.keys():

            u = u_s[mod_in][mod_in]
            uss.append(u)
            lpu = self.pu(*self.pu_params).log_prob(u).sum(-1)
            #lpu_klds.append(lpu)
            # print("lpu.shape")
            # print(lpu.shape)
            logs_kld_pu[mod_in] = lpu.clone().detach().sum()/u.size(0)

            w, z = torch.split(u, [self.latent_dim_w, self.latent_dim], dim=-1)

            lqz_x = log_mean_exp(torch.stack([qz_xs[qz_x_key].log_prob(z).sum(-1) for qz_x_key in qz_xs.keys()]))

            # print("lqz_x.shape")
            # print(lqz_x.shape)

            logs_kld_z[mod_in] = lqz_x.clone().detach().sum()/u.size(0)

            lqw_x = qw_xs[mod_in].log_prob(w).sum(-1)

            logs_kld_w[mod_in] = lqw_x.clone().detach().sum()/u.size(0)

          #  print("lqw_x.shape")
          #  print(lqw_x.shape)

            lpx_u =  self.compute_reconstruction_error( x, pxu[mod_in])

            # print("lpx_u.shape")
            # print(lpx_u.shape)

            logs_recon[mod_in] = lpx_u.clone().detach().sum()/u.size(0)
            #lpx_us.append(lpx_u)


            lw = lpx_u + self.beta*(lpu - lqz_x - lqw_x)

            # print("lw.shape")
            # print(lw.shape)

            lws.append(lw)
        
        lws= torch.stack(lws) 
        uss = torch.stack(uss)
        # print("lws.shape")
        # print(lws.shape)




        with torch.no_grad():
            grad_wt = (lws - torch.logsumexp(lws, 1, keepdim=True)).exp()
            if uss.requires_grad:
                uss.register_hook(lambda grad: grad_wt.unsqueeze(-1) * grad)
        return (grad_wt * lws).mean(0).sum(), [  logs_recon,logs_kld_pu ,logs_kld_z ,logs_kld_w ]

  
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), 
                                     lr=self.lr , amsgrad=True )
        return optimizer



   
  
   
  


   


    def compute_reconstruction_error(self, x, reconstruction):
        
        logprobs =[]
        for  idx, mod in enumerate( self.modalities_list ):
            #print(reconstruction[mod.name].shape)
           
            logprob = (mod.calc_log_prob( x[mod.name],reconstruction[mod.name],reduction=None )).view(*reconstruction[mod.name].size()[:2], -1).sum(-1)

            logprobs.append(logprob )
            logprobs[idx]*= float(mod.reconstruction_weight)  
              
        return torch.stack(logprobs).sum(0)
    



   
    def conditional_gen_all_subsets(self, x , N =None):
        self.eval()
        modalities_str = np.array([mod.name for mod in self.modalities_list])
     
        with torch.no_grad():
            encodings = self.encode(x)
            _, z_s = self.posterior.forward_w(encodings, self.pu,self.pu_params,self.latent_dim,self.latent_dim_w )   
            out = self.decode_train(z_s,flatten=True)
        return out 
    
    
    
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
    


class MixtureOfExpertsPlus(nn.Module):
    """Return parameters for product of independent experts as implemented in:
    See https://github.com/thomassutter/MoPoE

   
    
    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """

    def get_pw_params(self,_pw_params):
            return _pw_params[0], F.softmax(_pw_params[1], dim=1) * _pw_params[1].size(-1)
        
    def forward(self, encodings, pw ,pu, pw_param , latent_dim,latent_dim_w, K=1 ):
        
        post_experts = {}
        post_z = {}

        
        post_experts = {mod_in: {}  for mod_in in encodings.keys()}
        post_z = {mod_in: {}  for mod_in in encodings.keys()}
        
        for mod_in in encodings.keys():
            #print(mod_in)
            post_experts[mod_in][mod_in] = encodings[mod_in]
            post_z[mod_in][mod_in] = pu(*cat_shared_private( encodings[mod_in]) ).rsample(torch.Size([K]))
           
         
        for mod_in in encodings.keys():
            
            for mod_out in encodings.keys():
                
                if mod_in != mod_out:
                    post_experts[mod_in][mod_out] = {"shared": encodings[mod_in]["shared"],
                    "private" : None
                    }
                    u = post_z[mod_in][mod_in]
                    # print(mod_in)
                    # print(u.shape)
                    _,z_shared= torch.split(u, [latent_dim_w, latent_dim], dim=-1) 

                    z_private= pw(*self.get_pw_params(pw_param[mod_out] )).rsample(torch.Size([u.size()[0], u.size()[1]])).squeeze(2).type_as(z_shared)
                 
                    post_z[mod_in][mod_out] = torch.cat( [z_private,z_shared] ,dim=-1)

                    #for key in pw_param.keys():
                    #        pw_param[key][1].retain_grad()
        
        return post_experts, post_z
        


    def forward_w(self, encodings, pu, pu_params , latent_dim,latent_dim_w, K=1 ):
        
        post_experts = {}
        post_z = {}

            
        post_experts = {mod_in: {}  for mod_in in encodings.keys()}
        post_z = {mod_in: {}  for mod_in in encodings.keys()}
        
        for mod_in in encodings.keys():
                post_experts[mod_in][mod_in] = encodings[mod_in]
                post_z[mod_in][mod_in] = pu(*cat_shared_private( encodings[mod_in]) ).rsample(torch.Size([K]))
            
            
        for mod_in in encodings.keys():     
            for mod_out in encodings.keys():     
                if mod_in != mod_out:
                    
                    post_experts[mod_in][mod_out] = {"shared": encodings[mod_in]["shared"],
                    "private" : None
                    }
                    u = post_z[mod_in][mod_in]
                    _,z_shared= torch.split(u, [latent_dim_w, latent_dim], dim=-1) 

                    u_new = pu(*pu_params).rsample(torch.Size([u.size()[0], u.size()[1]])).squeeze(2).type_as(z_shared)
                    w_private,_ = torch.split(u_new, [latent_dim_w, latent_dim], dim=-1)

                    post_z[mod_in][mod_out] = torch.cat( [w_private,z_shared] ,dim=-1)

            
        return post_experts, post_z
        
    
    
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
    
    
 