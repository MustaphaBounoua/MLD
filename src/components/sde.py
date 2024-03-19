
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.components.importance_sampling import sample_vp_truncated_q
from tqdm import tqdm


class VP_SDE():
    def __init__(self,device, beta_min=0.1, beta_max=20, N = 1000,importance_sampling =True ,liklihood_weighting= False,
                 N_inpaint=250,jump_lenght=10,jump_n_sample = 10 ):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.N = N
        self.T = 1
        self.importance_sampling = importance_sampling
        self.liklihood_weighting = liklihood_weighting
        self.device = device
        self.t_epsilon =  0.001
        
        self.N_inpaint= N_inpaint
        self.dt_inp = self.T/self.N_inpaint
      
        self.repaint_times_schedule = np.array( get_schedule_jump(t_T=N_inpaint, n_sample=1,
                           jump_length=jump_lenght, jump_n_sample= jump_n_sample,
                           jump2_length=1, jump2_n_sample=1,
                           jump3_length=1, jump3_n_sample=1,
                           start_resampling=self.N_inpaint)) +1
        
        
    def beta_t(self,t):
        return self.beta_min + t * (self.beta_max - self.beta_min) 
    
   
    
    def marg_prob(self, t,x):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        
        log_mean_coeff = log_mean_coeff.to(self.device)
     
        mean = torch.exp(log_mean_coeff)
        std = torch.sqrt(1 - torch.exp(2 * log_mean_coeff))
        return mean * torch.ones_like(x).to(self.device), std.view(-1,1) * torch.ones_like(x).to(self.device)
    

    

    def train_step(self,data,score_net,eps = 1e-6):
        
        if self.importance_sampling:
           # t = self.sample_importance_weighted_time_for_likelihood((data.shape[0],), eps=eps).to(self.device)
           t = ( self.sample_debiasing_t(shape=(data.shape[0],)) ).to(self.device)
           #t= self.sample_debiasing_t(shape=(data.shape[0],)).to(self.device)
        else:
            ## uniform
            t = ( (self.T - eps) * torch.rand((data.shape[0],)) + eps ).to(self.device)
            
        t = t.view(data.shape[0],1)
        ## random z ## integral of the brownian noise
        z = torch.randn_like(data).to(self.device)
        ## Estimate P_t(x)
        ## integral of  drift and diffusion coefficient 
        mean, std = self.marg_prob(t,data)
        ## perturb data
     
        ##
        perturbed_data =  mean * data +  std * z
        #perturbed_data = mean + torch.matmul(std, z)
        ## compute the score
        f,g = self.sde(t)
        
        score = score_net(perturbed_data, t, sigma= std, weighted =False )
        
        if self.liklihood_weighting:
            loss = torch.square((score * std/g ) + z) /2
        else:
            loss = torch.square((score) + z)
        return loss
     

    def sample_debiasing_t(self, shape):
        """
        non-uniform sampling of t to debias the weight std^2/g^2
        the sampling distribution is proportional to g^2/std^2 for t >= t_epsilon
        for t < t_epsilon, it's truncated
        """
        return sample_vp_truncated_q(shape, self.beta_min, self.beta_max, t_epsilon=self.t_epsilon, T=self.T)


     
    def sde(self,t):
        return -0.5*self.beta_t(t), torch.sqrt(self.beta_t(t))
    
    
    def euler_step(self, x_t, t,dt, score_net ):
        
        
        time = t*torch.ones(x_t.shape[0]).view(-1,1).to(self.device)
        
        mean,std = self.marg_prob(t,x_t)
        with torch.no_grad():
            s = score_net(x_t,time.to(self.device),std)
        
        f,g = self.sde(t)
        
        x = x_t - dt*(f*x_t - (g**2) *s)  + g * torch.sqrt( dt).to(self.device)  * torch.randn_like(x_t)
      
        return x , t-dt

   
    def do_reverse_diffusion(self, score_net,x, t_step, mask ,x_0):
   
        t = (t_step/self.N_inpaint) * torch.ones(x.size(0)).to(self.device).view(x.size(0),1)
        
        ## diffuse availible modality
        mean,std = self.marg_prob(t,x)
        if t_step>1:
            noise = torch.randn_like(x_0)
        else:
            noise = 0.0
            
        x_0_t = x_0  * mean + std *  noise
        ## concatenante with missing modality 
        
        x_aux = (x_0_t * mask) + x * (1. - mask)
       
        ## score
        
             #print(x_aux)
        if self.liklihood_weighting:
                s = score_net(x_aux.float(),t,std, weighted = False ).detach()
        else:
                s = score_net(x_aux.float(),t,std ).detach()
     
        
     
        f,g = self.sde(t)
        
        if t_step>1:
            noise = torch.randn_like(x)
        else:
            noise = 0.0
        ## Euler step
        if self.liklihood_weighting:
            x_out = x_aux - self.dt_inp * (f * x_aux - ( 0.5 * g ) * s)  + g * torch.sqrt(torch.tensor(self.dt_inp).to(self.device) ) * noise
        else:
            x_out = x_aux - self.dt_inp * (f * x_aux - ( g ** 2 ) * s )  + g * torch.sqrt(torch.tensor(self.dt_inp).to(self.device) ) * noise
        
        x_out = x_out * (1. - mask) + x_0 * mask
        
        
        
        return x_out 
    
        
         
  
    def do_diffusion(self, x,t_step):
        t = (t_step/self.N_inpaint) * torch.ones(x.size(0)).to(self.device).view(x.size(0),1) 
        f,g = self.sde(t)
        x = x + self.dt_inp * f * x + g * torch.sqrt(torch.tensor(self.dt_inp).to(self.device) ) * torch.randn_like(x)
        return x 
    
    
            
    def mod_repaint(self, x , mask , score_net,debug = None ):
        x_inter = []
        i=0
        time_pairs = list(zip(self.repaint_times_schedule[:-1], self.repaint_times_schedule[1:]))
        x_0 = x
        x_t = x_0.clone()
        
        for t_last, t_cur in  time_pairs:
            #print("t_last step is "+ str(t_last) )
            if t_cur < t_last:
                # reversediffusion
                x_t = self.do_reverse_diffusion(score_net = score_net,
                                x = x_t, t_step = t_last, mask = mask , x_0 = x_0)
                if debug !=None and i % debug ==0 :
                    x_inter.append(x_t)
                #print("reverse : going to "+ str(t_cur) )
            else:
                x_t = self.do_diffusion(x_t, t_last  )

            i+=1
        if debug !=None:
            x_inter.append(x_t)
            return x_t, x_inter
        else:
            return x_t
   
   
   
   
   
   
    
    def euler_step_impaiting(self, x_t, t,dt, score_net , mask = None):
        """_summary_

        Args:
            x_t (_type_):  x_t which ist the concat of the different latent space 
                (non availible modalities are filler with a random gaussian)
            t (_type_): time
            dt (_type_): delta t
            score_net (_type_): takes as input xt, t, std return the normalized score score/std.
            mask (_type_, optional): Boolean mask [1,1,0,0] 1 is availible (non availible otherwise)

        Returns:
            _type_: X_(t-1)
        """        
        time = t*torch.ones(x_t.shape[0]).view(-1,1).to(self.device)
        
        ## diffuse availible modality
        mean,std = self.marg_prob(t,x_t)
        x_0 = x_t  * mean + std *  torch.randn_like(x_t).to(self.device) 
        ## concatenante with missing modality 
        
        x_aux = (x_0 * mask) + x_t * (1. - mask)
       
        ## score
        with torch.no_grad():
            s = score_net(x_aux,time,std).detach()
        
        f,g = self.sde(t)
        ## Euler step
        
        x = x_aux - dt*(f*x_aux - (g**2) *s)  + g * torch.sqrt(dt) * torch.randn_like(x_t)
  
        x = x * (1. - mask) + x_t * mask
        
        return x, t-dt



    def sample_euler(self,x_c, score_net):
   
        
        t = torch.Tensor([1.0]).to(self.device)
        dt = torch.Tensor(t/self.N).to(self.device)
        
        mean,std = self.marg_prob(t,x_c)
        x_c= x_c * mean + std *  torch.randn_like(x_c).to(self.device) 
       
        while t>0:
            x_c,t= self.euler_step(x_c,t,dt,score_net)
        return x_c 


    def modality_impainting(self, score_net,x, mask):
   
        
        t = torch.Tensor([1.0]).to(self.device)
        dt = torch.Tensor(t/self.N).to(self.device)
        x_c = x    
        while t>0:
            x_c,t= self.euler_step_impaiting(x_c,t,dt,score_net,mask = mask)
        return x_c 
            
        










def get_schedule_jump(t_T, n_sample, jump_length, jump_n_sample,
                      jump2_length=1, jump2_n_sample=1,
                      jump3_length=1, jump3_n_sample=1,
                      start_resampling=100000000):

    jumps = {}
    for j in range(0, t_T - jump_length, jump_length):
        jumps[j] = jump_n_sample - 1

    jumps2 = {}
    for j in range(0, t_T - jump2_length, jump2_length):
        jumps2[j] = jump2_n_sample - 1

    jumps3 = {}
    for j in range(0, t_T - jump3_length, jump3_length):
        jumps3[j] = jump3_n_sample - 1

    t = t_T
    ts = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if (
            t + 1 < t_T - 1 and
            t <= start_resampling
        ):
            for _ in range(n_sample - 1):
                t = t + 1
                ts.append(t)

                if t >= 0:
                    t = t - 1
                    ts.append(t)

        if (
            jumps3.get(t, 0) > 0 and
            t <= start_resampling - jump3_length
        ):
            jumps3[t] = jumps3[t] - 1
            for _ in range(jump3_length):
                t = t + 1
                ts.append(t)

        if (
            jumps2.get(t, 0) > 0 and
            t <= start_resampling - jump2_length
        ):
            jumps2[t] = jumps2[t] - 1
            for _ in range(jump2_length):
                t = t + 1
                ts.append(t)
            jumps3 = {}
            for j in range(0, t_T - jump3_length, jump3_length):
                jumps3[j] = jump3_n_sample - 1

        if (
            jumps.get(t, 0) > 0 and
            t <= start_resampling - jump_length
        ):
            jumps[t] = jumps[t] - 1
            for _ in range(jump_length):
                t = t + 1
                ts.append(t)
            jumps2 = {}
            for j in range(0, t_T - jump2_length, jump2_length):
                jumps2[j] = jump2_n_sample - 1

            jumps3 = {}
            for j in range(0, t_T - jump3_length, jump3_length):
                jumps3[j] = jump3_n_sample - 1

    ts.append(-1)

    return ts