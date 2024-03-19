
import torch
import random



def deconcat(z,mod_list,sizes):
    z_mods={}
    idx=0
    for i,mod in enumerate( mod_list):
        z_mods[mod] = z[:,idx:idx+ sizes[i] ]
        idx +=sizes[i]
    return z_mods

def concat_vect(encodings):
    z = torch.Tensor()
    for key in encodings.keys():
        z = z.to(encodings[key].device)
        z = torch.cat( [z, encodings[key]],dim = -1 )
    return z 

def unsequeeze_dict(data):
        for key in data.keys():
            if data[key].ndim == 1 :
                data[key]= data[key].view(data[key].size(0),1)
        return data


class VP_SDE():
    def __init__(self, 
                 beta_min=0.1, 
                 beta_max=20, 
                 N = 1000,
                 importance_sampling =True ,
                 liklihood_weighting= False,
                 nb_mod = 2
                ):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.N = N
        self.T = 1
        self.importance_sampling = importance_sampling
        self.liklihood_weighting = liklihood_weighting
        self.device = "cuda"
        self.nb_mod = nb_mod
        self.t_epsilon = 1e-3
        
        
        
        
    def beta_t(self,t):
        return self.beta_min + t * (self.beta_max - self.beta_min) 
    
   
    def sde(self,t):
        return -0.5*self.beta_t(t), torch.sqrt(self.beta_t(t))
    
    
    def marg_prob(self, t,x):
        ## return mean std of p(x(t))
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        
        log_mean_coeff = log_mean_coeff.to(self.device)
     
        mean = torch.exp(log_mean_coeff)
        std = torch.sqrt(1 - torch.exp(2 * log_mean_coeff))
        return mean * torch.ones_like(x).to(self.device), std.view(-1,1) * torch.ones_like(x).to(self.device)
    

    def sample(self, t, data, mods_list ):
        
        
        nb_mods = len(mods_list)
        self.device = t.device
        
        x_t_m = {}
        std_m = {}
        mean_m = {}
        z_m = {}
        g_m = {}
        
        for i ,mod in enumerate( mods_list ):
            x_mod = data[mod]
           
            z = torch.randn_like(x_mod).to(self.device)
            f,g = self.sde(t[:,i])
            
            mean_i, std_i = self.marg_prob(t[:,i].view(x_mod.shape[0],1),x_mod)

            std_m[mod]= std_i
            mean_m [mod] = mean_i
            z_m [mod] = z
            g_m[mod] = g
            x_t_m[mod] =  mean_i * x_mod +  std_i * z
    

        #score = - score_net(X_t,time = t , std = STD )

        #loss = torch.square((score * STD ) + Z).sum(0)
       
        return x_t_m,z_m, std_m ,g_m , mean_m


    
    
    def train_step(self,data,score_net, eps = 1e-5, d = 0.5 ):
        #data= unsequeeze_dict(data)
        x = concat_vect(data)

        mods_list = list(data.keys())
        mods_sizes = [data[key].size(1) for key in mods_list ]

        nb_mods = len(mods_list)

        if self.importance_sampling:
            t = ( (self.T - eps) * self.sample_debiasing_t(shape=(x.shape[0],1)) + eps ).to(self.device)
        else:
            t = ( (self.T - eps) * torch.rand((x.shape[0],1)) + eps ).to(self.device)

        t_n = t.expand((x.shape[0],nb_mods ) )

        learn_cond = (torch.bernoulli(torch.tensor([d] )) ==1.0)
        mask = [1,1]
        if learn_cond:
            subsets = [[0,1],[1,0]]
            i = random.randint(0, len(mods_list)-1 )
            mask = subsets[i]
            mask_time = torch.tensor( mask ).to(self.device).expand(t_n.size()) 
            t_n = t_n * mask_time

        x_t_m , z_m , std_m , g_m, mean_m  = self.sample(t= t_n, data= data,mods_list= mods_list)

        score = - score_net(concat_vect(x_t_m),t = t_n , std = None )
        
        weight = 1.0
        if learn_cond:
             
            score_m = deconcat(score,mods_list,mods_sizes)
            for idx,i in enumerate(mask):

                if i ==0:
                     ## all the benchmark has two equal size mods
                    dim_clean = score_m[mods_list[idx]].size(1)
                    z_m.pop(mods_list[idx])
                    score_m.pop(mods_list[idx])
                    
                else:
                    dim_diff = score_m[mods_list[idx]].size(1)
            weight += dim_clean/dim_diff
            score = concat_vect(score_m) 
            
        loss =  weight * torch.square( score + concat_vect(z_m) ).sum(1, keepdim=False)
      
        return loss
     










    def train_step_cond(self,data,score_net, eps = 1e-5, d = 0.5 ):
        #data= unsequeeze_dict(data)
        x = concat_vect(data)

        mods_list = list(data.keys())

        nb_mods = len(mods_list)

        if self.importance_sampling:
            t = ( (self.T - eps) * self.sample_debiasing_t(shape=(x.shape[0],1)) + eps ).to(self.device)
        else:
            t = ( (self.T - eps) * torch.rand((x.shape[0],1)) + eps ).to(self.device)

        t_n = t.expand((x.shape[0],nb_mods ) )

        
        mask = [1,0]

        x_t_m , z_m , std_m , g_m, mean_m  = self.sample(t= t_n, data= data,mods_list= mods_list)

        learn_cond = (torch.bernoulli(torch.tensor([d] )) ==1.0)
        if learn_cond:
            mask = [1,0]
            mask_time = torch.tensor( mask ).to(self.device).expand(t_n.size()) 
            t_n = t_n * mask_time + 1.0 * (1 - mask_time)
            #print(learn_cond)
            #print(t_n)
            x_t = concat_vect({"x":x_t_m["x"],
                 "y": data["y"] })
        else:
            mask = [1,0]
            mask_time = torch.tensor( mask ).to(self.device).expand(t_n.size()) 
            t_n = t_n * mask_time + 0.0 * (1 - mask_time)

            x_t = concat_vect({"x":x_t_m["x"],
                 "y": torch.zeros_like(data["y"]) })

            #print(learn_cond)
            #print(t_n)

        score = - score_net(x_t,t = t_n , std = None )
        weight = 1.0  
        loss =  weight * torch.square( score + z_m["x"]).sum(1, keepdim=False)
        return loss






    def sample_debiasing_t(self, shape):
        """
        non-uniform sampling of t to debias the weight std^2/g^2
        the sampling distribution is proportional to g^2/std^2 for t >= t_epsilon
        for t < t_epsilon, it's truncated
        """
        return sample_vp_truncated_q(shape, self.beta_min, self.beta_max, t_epsilon=self.t_epsilon, T=self.T)




    def euler_step(self, x_t, t,dt, score_net ):
        
        
        time = t * torch.ones( (x_t.shape[0],self.nb_mod)).to(self.device)

        mean,std = self.marg_prob(t,x_t)

        with torch.no_grad():
            s = - score_net(x_t,time,std).detach()
        
        f,g = self.sde(t)
        x = x_t - dt*(f*x_t - (g**2) *s)  + g * torch.sqrt( dt).to(self.device)  * torch.randn_like(x_t)
        return x , t-dt

   
    def euler_step_impaiting(self, x_t,x_0, t,dt, score_net , mask = None, subset = None):
             
        time = t  * torch.ones( (x_t.shape[0], self.nb_mod) ).to(self.device)
        
        ## diffuse availible modality
        mean,std = self.marg_prob(t,x_t)
        
        mask_time = torch.tensor([ 1. if i in subset else 0. for i in range(self.nb_mod)]).to(self.device).expand(time.size()) 
        time = time * (1. - mask_time)
        
        x_aux = (x_0 * mask) + x_t * (1. - mask)
       
        ## score
        with torch.no_grad():
            
            s = - score_net(x_aux,time,std).detach()
        
        f,g = self.sde(t)
        ## Euler step
        if t == 0.001:
            noise = 0
        else :
            noise = torch.randn_like(x_t)
        x = x_aux - dt*(f*x_aux - (g**2) *s)  + g * torch.sqrt(dt) * noise
  
        x = x * (1. - mask) + x_0 * mask
        
        return x  , t-dt



    
    def euler_step_c(self, x_t, t,dt, score_net ):
        
        
        time = t  * torch.ones( (x_t.shape[0], self.nb_mod) ).to(self.device)
        
        ## diffuse availible modality
        mean,std = self.marg_prob(t,x_t)
        
        mask_time = torch.tensor([1,0]).to(self.device).expand(time.size()) 
        time = time * mask_time + 0.0 * (1. - mask_time)
        
        x_aux = x_t
       
        ## score
        with torch.no_grad():
          
            s = - score_net(x_aux,time,None).detach()
            s = s /std[:,:s.size(1)]
        
        f,g = self.sde(t)
        ## Euler step
        if t == 0.001:
            noise = 0
        else :
            noise = torch.randn_like(s)
        x_t_up = x_aux[:,:s.size(1)]

        x = x_t_up - dt*(f*x_t_up - (g**2) *s)  + g * torch.sqrt(dt) * noise

        x = torch.cat([x_t_up,x_aux[:,s.size(1):]],dim=1)
     
        return x  , t-dt

    def euler_step_impaiting_c(self, x_t,x_0, t,dt, score_net , mask = None):
             
        time = t  * torch.ones( (x_t.shape[0], self.nb_mod) ).to(self.device)
        
        ## diffuse availible modality
        mean,std = self.marg_prob(t,x_t)
        
        mask_time = torch.tensor([1,0]).to(self.device).expand(time.size()) 
        time = time * mask_time + 1.0 * (1. - mask_time)
        
        x_aux = (x_0 * mask) + x_t * (1. - mask)
       
        ## score
        with torch.no_grad():
            
            s = - score_net(x_aux,time,None).detach()
            s = s /std[:,:s.size(1)]


        f,g = self.sde(t)
        ## Euler step
        if t == 0.001:
            noise = 0
        else :
            noise = torch.randn_like(s)

        x_t_up = x_aux[:,:s.size(1)]

        x = x_t_up - dt*(f*x_t_up - (g**2) *s)  + g * torch.sqrt(dt) * noise
  
        x = torch.cat([x_t_up,x_0[:,s.size(1):]],dim=1)
        
        return x  , t-dt




    def modality_inpainting_c(self, score_net,x, mask , subset):
        
        t = torch.Tensor([1.0]).to(self.device)
        t_ind = 1.0
        dt = torch.Tensor( t/self.N).to(self.device)
        x_c = x    
        while t_ind>0:
            x_c,t = self.euler_step_impaiting_c(x_t= x_c, x_0= x.clone(),t= t, dt = dt, score_net= score_net,mask = mask )
            t_ind = t_ind - dt
        return x_c 
    

    def modality_inpainting(self, score_net,x, mask , subset):
        
        t = torch.Tensor([1.0]).to(self.device)
        t_ind = 1.0
        dt = torch.Tensor( t/self.N).to(self.device)
        x_c = x    
        while t_ind>0:
            x_c,t = self.euler_step_impaiting(x_t= x_c, x_0= x.clone(),t= t, dt = dt, score_net= score_net,mask = mask ,subset = subset)
            t_ind = t_ind - dt
        return x_c 

    def sample_euler(self,x_c, score_net):
   
        
        t = torch.Tensor([1.0]).to(self.device)
        dt = torch.Tensor(t/self.N).to(self.device)
        
        mean,std = self.marg_prob(t,x_c)
        x_c= x_c * mean + std *  torch.randn_like(x_c).to(self.device) 
       
        while t>0:
            x_c,t= self.euler_step(x_c,t,dt,score_net)
        return x_c 
    
    
    def sample_euler_c(self,x_c, score_net):
   
        
        t = torch.Tensor([1.0]).to(self.device)
        dt = torch.Tensor(t/self.N).to(self.device)
        
        mean,std = self.marg_prob(t,x_c)
        x_c= x_c * mean + std *  torch.randn_like(x_c).to(self.device) 
       
        while t>0:
            x_c,t= self.euler_step_c(x_c,t,dt,score_net)
        return x_c 


def sample_vp_truncated_q(shape, beta_min, beta_max, t_epsilon, T):
    if isinstance(T, float) or isinstance(T, int):
        T = torch.Tensor([T]).float()
    u = torch.rand(*shape).to(T)
    vpsde = VariancePreservingTruncatedSampling(beta_min=0.1, beta_max=20., t_epsilon=t_epsilon)
    return vpsde.inv_Phi(u.view(-1), T).view(*shape)











import numpy as np
import torch
import torch
import torch.nn as nn

def log_standard_normal(x):
    return - 0.5 * x ** 2 - np.log(2 * np.pi) / 2


def sample_rademacher(shape):
    return (torch.rand(*shape).ge(0.5)).float() * 2 - 1


def sample_gaussian(shape):
    return torch.randn(*shape)


def sample_v(shape, vtype='rademacher'):
    if vtype == 'rademacher':
        return sample_rademacher(shape)
    elif vtype == 'normal' or vtype == 'gaussian':
        return sample_gaussian(shape)
    else:
        Exception(f'vtype {vtype} not supported')


Log2PI = float(np.log(2 * np.pi))


def log_normal(x, mean, log_var, eps=0.00001):
    z = - 0.5 * Log2PI
    return - (x - mean) ** 2 / (2. * torch.exp(log_var) + eps) - log_var / 2. + z


def exponential_CDF(t, lamb):
    return 1 - torch.exp(- lamb * t)


def sample_truncated_exponential(shape, lamb, T):
    """
    sample from q(t) prop lamb*exp(-lamb t) for t in [0, T]
    (one-sided truncation)
    """
    if lamb > 0:
        return - torch.log(1 - torch.rand(*shape).to(T) * exponential_CDF(T, lamb) + 1e-10) / lamb
    elif lamb == 0:
        return torch.rand(*shape).to(T) * T
    else:
        raise Exception(f'lamb must be nonnegative, got {lamb}')


def truncated_exponential_density(t, lamb, T):
    if lamb > 0:
        return lamb * torch.exp(-lamb * t) / exponential_CDF(T, lamb)
    elif lamb == 0:
        return 1 / T
    else:
        raise Exception(f'lamb must be nonnegative, got {lamb}')


def get_beta(iteration, anneal, beta_min=0.0, beta_max=1.0):
    assert anneal >= 1
    beta_range = beta_max - beta_min
    return min(beta_range * iteration / anneal + beta_min, beta_max)


def sample_ve_truncated_q(shape, sigma_min, sigma_max, t_epsilon, T):
    if isinstance(T, float) or isinstance(T, int):
        T = torch.Tensor([T]).float()
    u = torch.rand(*shape).to(T)
    return ve_truncated_q_inv_Phi(u.view(-1), sigma_min, sigma_max, t_epsilon, T).view(*shape)


def ve_truncated_q_density(t, sigma_min, sigma_max, t_epsilon, T):
    m = sigma_min ** 2
    r = (sigma_max / sigma_min) ** 2
    A1 = np.log(r) * t_epsilon * m * r ** t_epsilon / (m * r ** t_epsilon - m)
    A2 = (torch.log(m * r ** T - m) - np.log(m * r ** t_epsilon - m))
    A = A1 + A2

    gs2e = m * r ** t_epsilon * np.log(r) / (m * r ** t_epsilon - m) / A
    gs2 = m * r ** t * np.log(r) / (m * r ** t - m) / A

    return - torch.relu(- gs2 + gs2e) + gs2e


def ve_truncated_q_inv_Phi(u, sigma_min, sigma_max, t_epsilon, T):
    if isinstance(T, float) or isinstance(T, int):
        T = torch.Tensor([T]).float()
    # u = torch.rand(*shape).to(T)
    m = sigma_min ** 2
    r = (sigma_max / sigma_min) ** 2
    A1 = t_epsilon * m * r ** t_epsilon * np.log(r) / (m * r ** t_epsilon - m)
    A2 = (torch.log(m * r ** T - m) - np.log(m * r ** t_epsilon - m))
    A = A1 + A2

    # linear
    x_l = u * A * (m * r ** t_epsilon - m) / (m * r ** t_epsilon * np.log(r))
    mask = x_l.ge(t_epsilon).float()

    # nonlinear
    x_n = torch.log((r ** t_epsilon - 1) * torch.exp((A * u - A1)) + 1) / np.log(r)
    return mask * x_n + (1 - mask) * x_l


def ve_truncated_q_Phi(t, sigma_min, sigma_max, t_epsilon, T):
    if isinstance(T, float) or isinstance(T, int):
        T = torch.Tensor([T]).float()
    # u = torch.rand(*shape).to(T)
    m = sigma_min ** 2
    r = (sigma_max / sigma_min) ** 2
    A1 = t_epsilon * m * r ** t_epsilon * np.log(r) / (m * r ** t_epsilon - m)
    A2 = (torch.log(m * r ** T - m) - np.log(m * r ** t_epsilon - m))
    A = A1 + A2

    # linear
    u_l = t * (m * r ** t_epsilon * np.log(r)) / (A * (m * r ** t_epsilon - m))
    mask = t.ge(t_epsilon).float()

    # nonlinear
    u_n = A1 / A + (torch.log(m * r ** t - m) - np.log(m * r ** t_epsilon - m)) / A
    return mask * u_n + (1 - mask) * u_l


# noinspection PyTypeChecker
class VariancePreservingTruncatedSampling:

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20., t_epsilon=1e-3):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.t_epsilon = t_epsilon

    def beta(self, t):
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def integral_beta(self, t):
        return 0.5 * t ** 2 * (self.beta_max - self.beta_min) + t * self.beta_min

    def mean_weight(self, t):
        # return torch.exp( -0.25 * t**2 * (self.beta_max-self.beta_min) - 0.5 * t * self.beta_min )
        return torch.exp(-0.5 * self.integral_beta(t))

    def var(self, t):
        # return 1. - torch.exp( -0.5 * t**2 * (self.beta_max-self.beta_min) - t * self.beta_min )
        return 1. - torch.exp(- self.integral_beta(t))

    def std(self, t):
        return self.var(t) ** 0.5

    def g(self, t):
        beta_t = self.beta(t)
        return beta_t ** 0.5

    def r(self, t):
        return self.beta(t) / self.var(t)

    def t_new(self, t):
        mask_le_t_eps = (t <= self.t_epsilon).float()
        t_new = mask_le_t_eps * t_eps + (1. - mask_le_t_eps) * t
        return t_new

    def unpdf(self, t):
        t_new = self.t_new(t)
        unprob = self.r(t_new)
        return unprob

    def antiderivative(self, t):
        return torch.log(1. - torch.exp(- self.integral_beta(t))) + self.integral_beta(t)

    def phi_t_le_t_eps(self, t):
        t_eps = torch.tensor(float(self.t_epsilon))
        return self.r(t_eps).item() * t

    def phi_t_gt_t_eps(self, t):
        t_eps = torch.tensor(float(self.t_epsilon))
        return self.phi_t_le_t_eps(t_eps).item() + self.antiderivative(t) - self.antiderivative(t_eps).item()

    def normalizing_constant(self, T):
        return self.phi_t_gt_t_eps(T)

    def pdf(self, t, T):
        Z = self.normalizing_constant(T)
        prob = self.unpdf(t) / Z
        return prob

    def Phi(self, t, T):
        Z = self.normalizing_constant(T)
        t_new = self.t_new(t)
        mask_le_t_eps = (t <= self.t_epsilon).float()
        phi = mask_le_t_eps * self.phi_t_le_t_eps(t) + (1. - mask_le_t_eps) * self.phi_t_gt_t_eps(t_new)
        return phi / Z

    def inv_Phi(self, u, T):
        t_eps = torch.tensor(float(self.t_epsilon))
        Z = self.normalizing_constant(T)
        r_t_eps = self.r(t_eps).item()
        antdrv_t_eps = self.antiderivative(t_eps).item()
        mask_le_u_eps = (u <= self.t_epsilon * r_t_eps / Z).float()
        a = self.beta_max - self.beta_min
        b = self.beta_min
        inv_phi = mask_le_u_eps * Z / r_t_eps * u + (1. - mask_le_u_eps) * \
                  (-b + (b ** 2 + 2. * a * torch.log(
                      1. + torch.exp(Z * u + antdrv_t_eps - r_t_eps * self.t_epsilon))) ** 0.5) / a
        return inv_phi


# noinspection PyUnusedLocal
def sample_vp_truncated_q(shape, beta_min, beta_max, t_epsilon, T):
    if isinstance(T, float) or isinstance(T, int):
        T = torch.Tensor([T]).float()
    u = torch.rand(*shape).to(T)
    vpsde = VariancePreservingTruncatedSampling(beta_min=0.1, beta_max=20., t_epsilon=t_epsilon)
    return vpsde.inv_Phi(u.view(-1), T).view(*shape)


# noinspection PyUnusedLocal
def get_normalizing_constant(shape,T =1.0):
    if isinstance(T, float) or isinstance(T, int):
        T = torch.Tensor([T]).float()
    u = torch.rand(*shape).to(T)
    vpsde = VariancePreservingTruncatedSampling(beta_min=0.1, beta_max=20.0, t_epsilon=0.001)
    return vpsde.normalizing_constant(T=T)






Log2PI = float(np.log(2 * np.pi))

def log_normal(x, mean, log_var, eps=0.00001):
    z = - 0.5 * Log2PI
    return - (x - mean) ** 2 / (2. * torch.exp(log_var) + eps) - log_var / 2. + z

def sample_rademacher(shape):
    return (torch.rand(*shape).ge(0.5)).float() * 2 - 1

def sample_gaussian(shape):
    return torch.randn(*shape)

def sample_v(shape, vtype='rademacher'):
    if vtype == 'rademacher':
        return sample_rademacher(shape)
    elif vtype == 'normal' or vtype == 'gaussian':
        return sample_gaussian(shape)
    else:
        Exception(f'vtype {vtype} not supported')

class VariancePreservingSDE(torch.nn.Module):
    """
    Implementation of the variance preserving SDE proposed by Song et al. 2021
    See eq (32-33) of https://openreview.net/pdf?id=PxTIG12RRHS
    """
    def __init__(self, beta_min=0.1, beta_max=20.0, T=1.0, t_epsilon=0.001):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
        self.t_epsilon = t_epsilon

    @property
    def logvar_mean_T(self):
        logvar = torch.zeros(1)
        mean = torch.zeros(1)
        return logvar, mean

    def beta(self, t):
        return self.beta_min + (self.beta_max-self.beta_min)*t

    def mean_weight(self, t):
        return torch.exp(-0.25 * t**2 * (self.beta_max-self.beta_min) - 0.5 * t * self.beta_min)
    
    def log_mean_weight(self,t):
        return (-0.25 * t**2 * (self.beta_max-self.beta_min) - 0.5 * t * self.beta_min)


    def var(self, t):
        return 1. - torch.exp(-0.5 * t**2 * (self.beta_max-self.beta_min) - t * self.beta_min)

    def f(self, t, y):
        return - 0.5 * self.beta(t) * y

    def g(self, t, y):
        beta_t = self.beta(t)
        return torch.ones_like(y) * beta_t**0.5

    def sample(self, t, y0, return_noise=False):
        """
        sample yt | y0
        if return_noise=True, also return std and g for reweighting the denoising score matching loss
        """
        
        mu = self.mean_weight(t) * y0
        std = self.var(t) ** 0.5
        epsilon = torch.randn_like(y0)
        yt = epsilon * std + mu
        if not return_noise:
            return yt
        else:
            return yt, epsilon, std, self.g(t, yt)


    def sample_image(self, t, y0, return_noise=False):
        """
        sample yt | y0
        if return_noise=True, also return std and g for reweighting the denoising score matching loss
        """
        #t_ = t.view(t.size(0) ) 
        mu_ = self.mean_weight(t)
        
        mu = mu_.view(t.size(0),1,1,1).expand(y0.shape) * y0

        
        std_ = self.var(t)

        std = std_.view(t.size(0),1,1,1).expand(y0.shape) ** 0.5

        epsilon = torch.randn_like(y0)

        yt = epsilon * std + mu
        if not return_noise:
            return yt
        else:
            return yt, epsilon, std, self.g(t.view(t.size(0),1,1,1), yt)


    

    def sample_debiasing_t(self, shape,T = None):
        """
        non-uniform sampling of t to debias the weight std^2/g^2
        the sampling distribution is proportional to g^2/std^2 for t >= t_epsilon
        for t < t_epsilon, it's truncated
        """
        if T == None:
            T = self.T
        return sample_vp_truncated_q(shape, self.beta_min, self.beta_max, t_epsilon=self.t_epsilon, T=T)



class PluginReverseSDE(torch.nn.Module):
    """
    inverting a given base sde with drift `f` and diffusion `g`, and an inference sde's drift `a` by
    f <- g a - f
    g <- g
    (time is inverted)
    """
    def __init__(self, base_sde, T, vtype='rademacher', debias=True):
        super().__init__()
        self.base_sde = base_sde
       
        self.T = T
        self.vtype = vtype
        self.debias = debias

    # Drift
    def mu(self, t, y, lmbd=0.):
        return (1. - 0.5 * lmbd) * self.base_sde.g(self.T-t, y) * self.a(y, self.T - t.squeeze()) - \
               self.base_sde.f(self.T - t, y)

    # Diffusion
    def sigma(self, t, y, lmbd=0.):
        return (1. - lmbd) ** 0.5 * self.base_sde.g(self.T-t, y)

    @torch.enable_grad()
    def dsm(self, x, a , weight =False, eps = 1e-6):
        """
        denoising score matching loss
        """
        if self.debias:
            t_ = (1.0-eps) * self.base_sde.sample_debiasing_t([x.size(0), ] + [1 for _ in range(x.ndim - 1)]) + eps
        else:
            t_ =  (1.0-eps) * torch.rand(([x.size(0), 1])) + eps 
            
        t_ = t_.to(x.device) 
        y, target, std, g = self.base_sde.sample(t_, x, return_noise=True)
        
        if weight==False:
            s = - a(y, t_,std)
            loss= ((s * std  + target) ** 2).view(x.size(0), -1).sum(1, keepdim=False) 
        else:
            s = a(y, t_,None)
            loss=((s * std / g + target) ** 2).view(x.size(0), -1).sum(1, keepdim=False) / 2
    

        return loss

    
    



    def elbo_random_t_slice(self, x,a):
        """
        estimating the ELBO of the plug-in reverse SDE by sampling t uniformly between [0, T], and by estimating
        div(mu) using the Hutchinson trace estimator
        """
        t_ = torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(x) * self.T
        qt = 1 / self.T
        y = self.base_sde.sample(t_, x).requires_grad_()

        a = a(y, t_.squeeze())
        mu = self.base_sde.g(t_, y) * a - self.base_sde.f(t_, y)

        v = sample_v(x.shape, vtype=self.vtype).to(y)

        Mu = - (
              torch.autograd.grad(mu, y, v, create_graph=self.training)[0] * v
        ).view(x.size(0), -1).sum(1, keepdim=False) / qt

        Nu = - (a ** 2).view(x.size(0), -1).sum(1, keepdim=False) / 2 / qt
        yT = self.base_sde.sample(torch.ones_like(t_) * self.base_sde.T, x)
        lp = log_normal(yT, torch.zeros_like(yT), torch.zeros_like(yT)).view(x.size(0), -1).sum(1)

        return lp + Mu + Nu


 
    





   