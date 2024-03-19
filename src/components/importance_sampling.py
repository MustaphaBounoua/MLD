import numpy as np
import torch

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
        t_new = mask_le_t_eps * self.t_epsilon + (1. - mask_le_t_eps) * t
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
    vpsde = VariancePreservingTruncatedSampling(beta_min=beta_min, beta_max=beta_max,t_epsilon=t_epsilon)
    return vpsde.inv_Phi(u.view(-1), T).view(*shape)

if __name__ =="__main__" :
    for i in range(2000000):
        t = sample_vp_truncated_q([2,2], 0.1, 20, 1e-3, 1.0)
        
        if torch.sum(t < 1e-3) > 1:
            print(1/t)