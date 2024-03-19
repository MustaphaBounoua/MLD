import math
from functools import partial
from src.components.attention import MultiheadAttention
import torch
from einops import rearrange
from src.components.attention import MultiheadAttention, PreNorm
import torch.nn as nn

# constants

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t



def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        #nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Linear(dim, default(dim_out, dim))
    )

def Downsample(dim, dim_out = None):
    return nn.Linear(dim, default(dim_out, dim))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim ):
        super().__init__()
        self.dim = dim
       

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
    
class SinusoidalPosEmbMultimodal(nn.Module):
    def __init__(self, dim,nb_mod ):
        super().__init__()
        self.dim = dim
        self.nb_mod =nb_mod
        self.sinusoidalPosEmb = SinusoidalPosEmb(dim=dim)
    def forward(self,x):
        out = []
        for i in range(self.nb_mod): 
            out.append( self.sinusoidalPosEmb(x[:,i] ).view( x.size(0),self.dim ) ) 
        return torch.cat(out,-1)
        
        
        
            

   







class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8,shift_scale =True):
        super().__init__()
        #self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.proj = nn.Linear(dim,dim_out)
        self.act = nn.SiLU()
        self.norm = nn.GroupNorm(groups, dim)
       # self.norm = nn.BatchNorm1d( dim_out)
        self.shift_scale = shift_scale
        

    def forward(self, x, t = None ):
        x = self.norm(x)
        x = self.act(x)
        x = self.proj(x)
        
        if exists(t):
            if self.shift_scale:
                scale, shift = t
                x = x * (scale.squeeze() + 1) + shift.squeeze()
            else:
                x = x + t
        
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8, shift_scale = True):
        super().__init__()
        self.shift_scale= shift_scale
        self.mlp = nn.Sequential(
            nn.SiLU(),
            #nn.Linear(time_emb_dim, dim_out * 2)
            nn.Linear(time_emb_dim, dim_out*2 if shift_scale else dim_out)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups, shift_scale = shift_scale)
        self.block2 = Block(dim_out, dim_out, groups = groups,shift_scale = shift_scale)
        #self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.lin_layer = nn.Linear(dim, dim_out) if dim != dim_out else nn.Identity()
        
    
    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            
            time_emb = self.mlp(time_emb)
            if self.shift_scale:
                time_emb = rearrange(time_emb, 'b c -> b c 1')
                scale_shift = time_emb.chunk(2, dim = 1)
            else:
                scale_shift = time_emb 
      
        h = self.block1(x, t = scale_shift)

        h = self.block2(h)

        return h + self.lin_layer(x)








from src.utils import deconcat, concat_vect

class UnetMLP(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        resnet_block_groups = 8,
        time_dim = 96,
        nb_mod = 1,
        use_attention = True,
        num_head = 2,
        shift_scale = True,
        dim_head = None,
        modalities = None
    ):
        super().__init__()

        # determine dimensions
        self.modalities_list = modalities

        self.use_attention = use_attention
        init_dim = default(init_dim, dim)
        if init_dim == None:
            init_dim = dim * dim_mults[0] 
        self.init_lin= nn.Linear(dim, init_dim)

        dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

       
        
        block_klass = partial(ResnetBlock, groups = resnet_block_groups,shift_scale = shift_scale)
        
    
        # time embeddings
        sinus_pos_embed =  SinusoidalPosEmbMultimodal(init_dim//2,nb_mod)
        
        # fourier_dim = dim//2 * nb_mod
        fourier_dim = init_dim
        # self.time_mlp = nn.Sequential(
        #     sinus_pos_embed,
        #     nn.Linear(fourier_dim, time_dim),
        #     nn.GELU(),
        #     nn.Linear(time_dim, time_dim)
        # )

        self.time_mlp = nn.Sequential(
            nn.Linear(nb_mod,time_dim),
           # nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            module = nn.ModuleList([block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                            #        block_klass(dim_in, dim_in, time_emb_dim = time_dim)
                      ])
            
            if use_attention:
                module.append(
                   # TransformerEncoderLayer(size=dim_in,ff_size = dim_in, num_heads= num_head, d=0.0 )
                   Residual(PreNorm(dim_in,MultiheadAttention(dim_in,num_head, dim_head= dim_head)) )
                    )
            module.append( Downsample(dim_in, dim_out) if not is_last else nn.Linear(dim_in, dim_out))
            self.downs.append(module)


        mid_dim = dims[-1]
        joint_dim = mid_dim
       # joint_dim = 24
        self.mid_block1 = block_klass(mid_dim, joint_dim, time_emb_dim = time_dim)
        #self.mid_attn = Residual(PreNorm(mid_dim, MultiheadAttention(mid_dim,dim_head=mid_dim//2)))
        if use_attention:
            # self.mid_attn =  TransformerEncoderLayer(size=mid_dim,ff_size = mid_dim, num_heads= num_head, d=0.0 )
            self.mid_attn =  Residual( PreNorm(mid_dim,MultiheadAttention(mid_dim,num_head, dim_head= dim_head)) )
        self.mid_block2 = block_klass(joint_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            module = nn.ModuleList([ block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
               #       block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim) 
                      ])
            
            if use_attention:
                 module.append( 
                   #  TransformerEncoderLayer(size=dim_out,ff_size = dim_out,num_heads= num_head, d=0.0 )
                    Residual( PreNorm(dim_out,MultiheadAttention(dim_out,num_head , dim_head= dim_head)) )
                     )
            module.append( Upsample(dim_out, dim_in) if not is_last else  nn.Linear(dim_out, dim_in))
            self.ups.append(module)
                        
      

        # default_out_dim = channels * (1 if not learned_variance else 2)
       
        self.out_dim = default(out_dim, dim_in)

        self.final_res_block = block_klass(init_dim * 2, init_dim, time_emb_dim = time_dim)

        self.proj = nn.Linear(init_dim, dim)

        self.proj.weight.data.fill_(0.0)
        self.proj.bias.data.fill_(0.0)

        self.final_lin = nn.Sequential( 
            nn.GroupNorm(resnet_block_groups,init_dim),
            nn.SiLU(),
            self.proj
        )



    def forward(self, x, time, sigma , weighted = True , get_bottelneck = False):

        # x = x / x.std(dim=0)
        # x_mod = deconcat(x,modalities_list=self.modalities_list)
        # for key in x_mod.keys():
        #     x_mod[key] = x_mod[key]/ x_mod[key].std(axis=1, keepdim = True) 
        # x = concat_vect(x_mod)

        x = self.init_lin(x)
        
        r = x.clone()

        t = self.time_mlp(time).squeeze()
        
        h = []

        for blocks in self.downs:
            if self.use_attention:
               # block1,block2, attn, downsample = blocks
                block1, attn, downsample = blocks
            else:
              #  block1,block2, downsample = blocks
                block1, downsample = blocks
                
            
            x = block1(x, t)
            
            # h.append(x)
            # x = block2(x, t)
            
            if self.use_attention:
                x = attn(x)
            h.append(x)
            x = downsample(x)
      
        x = self.mid_block1(x, t)
        bottelneck = x.clone().detach()
        if self.use_attention:
            x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        
        for blocks in self.ups:
            
            if self.use_attention:
               # block1,block2, attn, upsample = blocks
                block1, attn, upsample = blocks
            else:
              #  block1,block2, upsample = blocks
                block1, upsample = blocks
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            # x = torch.cat((x, h.pop()), dim = 1)
            # x = block2(x, t)
         
            if self.use_attention:
                x = attn(x)
            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        if get_bottelneck == False:
            if weighted == True:
                return self.final_lin(x) /sigma
            else:
                return self.final_lin(x)
        else:
            if weighted == True:
                return self.final_lin(x) /sigma , bottelneck
            else:
                return self.final_lin(x), bottelneck
            



        
       
    



















# import math
# from random import random
# from functools import partial
# from collections import namedtuple
# from src.components.attention import MultiheadAttention
# import torch
# from torch import nn, einsum
# import torch.nn.functional as F

# from einops import rearrange
# from einops.layers.torch import Rearrange
# from src.components.attention import MultiheadAttention, PreNorm
# from tqdm.auto import tqdm
# from src.components.transformer import TransformerEncoderLayer
# # constants

# # helpers functions

# def exists(x):
#     return x is not None

# def default(val, d):
#     if exists(val):
#         return val
#     return d() if callable(d) else d

# def identity(t, *args, **kwargs):
#     return t



# def num_to_groups(num, divisor):
#     groups = num // divisor
#     remainder = num % divisor
#     arr = [divisor] * groups
#     if remainder > 0:
#         arr.append(remainder)
#     return arr


# # small helper modules

# class Residual(nn.Module):
#     def __init__(self, fn):
#         super().__init__()
#         self.fn = fn

#     def forward(self, x, *args, **kwargs):
#         return self.fn(x, *args, **kwargs) + x

# def Upsample(dim, dim_out = None):
#     return nn.Sequential(
#         #nn.Upsample(scale_factor = 2, mode = 'nearest'),
#         nn.Linear(dim, default(dim_out, dim))
#     )

# def Downsample(dim, dim_out = None):
#     return nn.Linear(dim, default(dim_out, dim))


# class Residual(nn.Module):
#     def __init__(self, fn):
#         super().__init__()
#         self.fn = fn

#     def forward(self, x, *args, **kwargs):
#         return self.fn(x, *args, **kwargs) + x


# # sinusoidal positional embeds

# class SinusoidalPosEmb(nn.Module):
#     def __init__(self, dim ):
#         super().__init__()
#         self.dim = dim
       

#     def forward(self, x):
#         device = x.device
#         half_dim = self.dim // 2
#         emb = math.log(10000) / (half_dim - 1)
#         emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
#         emb = x[:, None] * emb[None, :]
#         emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
#         return emb
    
    
# class SinusoidalPosEmbMultimodal(nn.Module):
#     def __init__(self, dim,nb_mod ):
#         super().__init__()
#         self.dim = dim
#         self.nb_mod =nb_mod
#         self.sinusoidalPosEmb = SinusoidalPosEmb(dim=dim)
#     def forward(self,x):
#         out = []
#         for i in range(self.nb_mod): 
#             out.append( self.sinusoidalPosEmb(x[:,i] ).view( x.size(0),self.dim ) ) 
#         return torch.cat(out,-1)
        
        
        
            

   







# class Block(nn.Module):
#     def __init__(self, dim, dim_out, groups = 8,shift_scale =True):
#         super().__init__()
#         #self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
#         self.proj = nn.Linear(dim,dim_out)
#         self.act = nn.SiLU()
#         self.norm = nn.GroupNorm(groups, dim_out)
#         self.shift_scale = shift_scale
        

#     def forward(self, x, t = None ):
#         x = self.proj(x)
#         x = self.norm(x)

#         if exists(t):
#             if self.shift_scale:
#                 scale, shift = t
#                 x = x * (scale.squeeze() + 1) + shift.squeeze()
#             else:
#                 x = x + t
#         x = self.act(x)
#         return x


# class ResnetBlock(nn.Module):
#     def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8, shift_scale = True):
#         super().__init__()
#         self.shift_scale= shift_scale
#         self.mlp = nn.Sequential(
#             nn.SiLU(),
#             #nn.Linear(time_emb_dim, dim_out * 2)
#             nn.Linear(time_emb_dim, dim_out*2 if shift_scale else dim_out)
#         ) if exists(time_emb_dim) else None

#         self.block1 = Block(dim, dim_out, groups = groups, shift_scale = shift_scale)
#         self.block2 = Block(dim_out, dim_out, groups = groups,shift_scale = shift_scale)
#         #self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
#         self.lin_layer = nn.Linear(dim, dim_out) if dim != dim_out else nn.Identity()
        
    
#     def forward(self, x, time_emb = None):

#         scale_shift = None
#         if exists(self.mlp) and exists(time_emb):
            
#             time_emb = self.mlp(time_emb)
#             if self.shift_scale:
#                 time_emb = rearrange(time_emb, 'b c -> b c 1')
#                 scale_shift = time_emb.chunk(2, dim = 1)
#             else:
#                 scale_shift = time_emb 
      
#         h = self.block1(x, t = scale_shift)

#         h = self.block2(h)

#         return h + self.lin_layer(x)








# from src.utils import deconcat, concat_vect

# class UnetMLP(nn.Module):
#     def __init__(
#         self,
#         dim,
#         init_dim = None,
#         out_dim = None,
#         dim_mults=(1, 2, 4, 8),
#         resnet_block_groups = 8,
#         time_dim = 96,
#         nb_mod = 1,
#         use_attention = True,
#         num_head = 2,
#         shift_scale = True,
#         dim_head = None,
#         modalities = None
#     ):
#         super().__init__()

#         # determine dimensions
#         self.modalities_list = modalities

#         self.use_attention = use_attention
#         init_dim = default(init_dim, dim)
#         self.init_lin= nn.Linear(dim, init_dim)

#         dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
#         in_out = list(zip(dims[:-1], dims[1:]))

       
        
#         block_klass = partial(ResnetBlock, groups = resnet_block_groups,shift_scale = shift_scale)

#         # time embeddings
#         sinus_pos_embed =  SinusoidalPosEmbMultimodal(dim//2,nb_mod)
        
#         # fourier_dim = dim//2 * nb_mod
#         fourier_dim = dim
#         # self.time_mlp = nn.Sequential(
#         #     sinus_pos_embed,
#         #     nn.Linear(fourier_dim, time_dim),
#         #     nn.GELU(),
#         #     nn.Linear(time_dim, time_dim)
#         # )

#         self.time_mlp = nn.Sequential(
#             nn.Linear(nb_mod,fourier_dim),
#             nn.Linear(fourier_dim, time_dim),
#             nn.GELU(),
#             nn.Linear(time_dim, time_dim)
#         )

#         # layers

#         self.downs = nn.ModuleList([])
#         self.ups = nn.ModuleList([])
        
#         num_resolutions = len(in_out)

#         for ind, (dim_in, dim_out) in enumerate(in_out):
#             is_last = ind >= (num_resolutions - 1)

#             module = nn.ModuleList([block_klass(dim_in, dim_in, time_emb_dim = time_dim),
#                                     block_klass(dim_in, dim_in, time_emb_dim = time_dim)
#                       ])
            
#             if use_attention:
#                 module.append(
#                    # TransformerEncoderLayer(size=dim_in,ff_size = dim_in, num_heads= num_head, d=0.0 )
#                    Residual(PreNorm(dim_in,MultiheadAttention(dim_in,num_head, dim_head= dim_head)) )
#                     )
#             module.append( Downsample(dim_in, dim_out) if not is_last else nn.Linear(dim_in, dim_out))
#             self.downs.append(module)


#         mid_dim = dims[-1]
        
#         self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
#         #self.mid_attn = Residual(PreNorm(mid_dim, MultiheadAttention(mid_dim,dim_head=mid_dim//2)))
#         if use_attention:
#             # self.mid_attn =  TransformerEncoderLayer(size=mid_dim,ff_size = mid_dim, num_heads= num_head, d=0.0 )
#             self.mid_attn =  Residual( PreNorm(mid_dim,MultiheadAttention(mid_dim,num_head, dim_head= dim_head)) )

#         self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

#         for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
#             is_last = ind == (len(in_out) - 1)
#             module = nn.ModuleList([ block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
#                       block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim) 
#                       ])
            
#             if use_attention:
#                  module.append( 
#                    #  TransformerEncoderLayer(size=dim_out,ff_size = dim_out,num_heads= num_head, d=0.0 )
#                     Residual( PreNorm(dim_out,MultiheadAttention(dim_out,num_head , dim_head= dim_head)) )
#                      )
#             module.append( Upsample(dim_out, dim_in) if not is_last else  nn.Linear(dim_out, dim_in))
#             self.ups.append(module)
                        
      

#         # default_out_dim = channels * (1 if not learned_variance else 2)
       
#         self.out_dim = default(out_dim, dim_in)

#         self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
#         self.final_lin = nn.Linear(dim, self.out_dim)

#     def forward(self, x, time, sigma , weighted = True):

#         # x = x / x.std(dim=0)
#         # x_mod = deconcat(x,modalities_list=self.modalities_list)
#         # for key in x_mod.keys():
#         #     x_mod[key] = x_mod[key]/ x_mod[key].std(axis=1, keepdim = True) 
#         # x = concat_vect(x_mod)

#         x = self.init_lin(x)
        
#         r = x.clone()

#         t = self.time_mlp(time).squeeze()
        
#         h = []

#         for blocks in self.downs:
#             if self.use_attention:
#                 block1,block2, attn, downsample = blocks
#             else:
#                 block1,block2, downsample = blocks
                
            
#             x = block1(x, t)
            
#             h.append(x)
#             x = block2(x, t)
            
#             if self.use_attention:
#                 x = attn(x)
#             h.append(x)
#             x = downsample(x)
      
#         x = self.mid_block1(x, t)
#         if self.use_attention:
#             x = self.mid_attn(x)
#         x = self.mid_block2(x, t)

#         for blocks in self.ups:
            
#             if self.use_attention:
#                 block1,block2, attn, upsample = blocks
#             else:
#                 block1,block2, upsample = blocks
            
#             x = torch.cat((x, h.pop()), dim = 1)
#             x = block1(x, t)

#             x = torch.cat((x, h.pop()), dim = 1)
#             x = block2(x, t)
         
#             if self.use_attention:
#                 x = attn(x)
#             x = upsample(x)

#         x = torch.cat((x, r), dim = 1)

#         x = self.final_res_block(x, t)
#         if weighted == True:
#             return self.final_lin(x) /sigma
#         else:
#             return self.final_lin(x)
            



        
       
    