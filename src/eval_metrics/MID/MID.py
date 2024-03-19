"""
    mid.metric 
    Copyright (c) 2022-present NAVER Corp.
    Apache-2.0
"""

from typing import *
import clip
from torch.nn import Module
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from src.eval_metrics.MID.metrics import *

def escape(x):
    return x.replace('-', '_').replace('/', '_')


def get_clip(eval_model: Module, device: Union[torch.device, int]) \
        -> Tuple[Module, Module]:
    """Get the CLIP model

    Args:
        eval_model (Module): The CLIP model to evaluate
        device (Union[torch.device, int]): Device index to select

    Returns:
        Tuple[Module, Module]: The CLIP model and a preprocessor
    """
    clip_model, _ = clip.load(eval_model)
    clip_model = clip_model.to(device)
    clip_model = clip_model.eval()
    clip_prep = T.Compose([T.Resize(224),
                           T.Normalize((0.48145466, 0.4578275, 0.40821073),
                                       (0.26862954, 0.26130258, 0.27577711))])
    return clip_model, clip_prep


def init_metric_list (eval_model,limit,device,add_mid):
    
    METRICS = [ #MutualInformationDivergence,  # Ours
                ClipScore,                    # CLIP-S
              #  RPrecision,                   # CLIP-R-Precision
                ]
    if add_mid :
        METRICS.append(MutualInformationDivergence)
    return [init_metric( x, eval_model, limit, device) for x in METRICS]

def init_metric(metric, eval_model,
                limit, device) :
    """Initialize a given metric class.

    Args:
        root (str): Path to data directory
        metric (Type[Metric]): Metric class
        eval_model (Module): Evaluating CLIP model
        limit (int, optional): Number of reference samples
        device (torch.device): Device index to select

    Returns:
        Metric: Initialized metric instance
    """
    m = metric(768 if eval_model == 'ViT-L/14' else 512,
                   limit=limit)
    m.cuda(device)
    m._debug = False
    return m



@torch.no_grad()
def populate_metrics_step(metrics, image, text , clip_model,clip_prep, modalities_list,img_ref =None, text_ref=None):
        """Populate the list of metrics using a given data loader.

        Args:
            dataloader (DataLoader): Data loader
            metrics (List[Metric]): List of metrics
            clip_model (Module): Evaluating CLIP model

        Returns:
            Tensor: Labels
        """
       

        device = next(clip_model.parameters()).device
       

        image = image.to(device)
        text = modalities_list[1].get_str( text )
   
        #image_gen = data["image"].to(device)
        
   
        
        txt = clip.tokenize(text, truncate=True).to(device)
     
        
            
        txt_features = clip_model.encode_text(txt).float()
        image_features = clip_model.encode_image(clip_prep(image)).float()
        
        txt_features = F.normalize(txt_features, dim=-1)
        image_features = F.normalize(image_features, dim=-1)
        
        #fake_im_features = clip_model.encode_image(clip_prep(image_gen)).float()

        # float16 of CLIP may suffer in l2-normalization
        #if text_ref ==None and img_ref == None: 
        x_gen = image_features
        y_ref = txt_features
        ## whatever we want here
        x_ref = x_gen
        
        if text_ref !=None:
            ## generating text from images
            text_ref = modalities_list[1].get_str( text_ref )
            text_ref = clip.tokenize(text_ref, truncate=True).to(device)
            text_ref_features = clip_model.encode_text(text_ref).float()
            text_ref_features = F.normalize(text_ref_features, dim=-1)
            
            x_gen = txt_features
            y_ref = image_features
            ##ground truth
            x_ref = text_ref_features
  
        if img_ref !=None:
            ## generating images from text
            img_ref=img_ref.to(device)
            img_ref_feature = clip_model.encode_image(clip_prep(img_ref)).float()
            img_ref_feature = F.normalize(img_ref_feature, dim=-1)
            x_ref = img_ref_feature
            
            
       
        
        #fake_im_features = F.normalize(fake_im_features, dim=-1)

        
        ##placeholder only
        
        
        for idx, m in enumerate(metrics):
            m.update(x_ref, y_ref, x_gen)
