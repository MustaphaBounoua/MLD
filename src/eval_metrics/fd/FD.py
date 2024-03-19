"""
    mid.metric 
    Copyright (c) 2022-present NAVER Corp.
    Apache-2.0
"""

from typing import *
import torch
from src.eval_metrics.fid.inception  import InceptionV3
from src.eval_metrics.fd.Frechet_distance import FrechetDistance
INCEPTION_FILE ="data/pt_inception-2015-12-05-6726825d.pth"

Device ="cuda:0"

def get_inception_net(dims = 2048, device = Device):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx], path_state_dict=INCEPTION_FILE)
    model = model.to(device)
    model = model.eval()
    return model

def init_metric_fd(act_dim,limit, device=Device) :
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
    device=Device
    m = FrechetDistance(act_dim,limit=limit)
   # m.cuda(device)
    m._debug = False
    return m

def preprocess_inception(x):
    return torch.clamp(x,max=1.0,min = 0.0).type(torch.FloatTensor)
    

@torch.no_grad()
def populate_metrics_step_fid(metric, x, y , classifier=None, inception_model=None,device=None):
        """Populate the list of metrics using a given data loader.

        Args:
            dataloader (DataLoader): Data loader
            metrics (List[Metric]): List of metrics
            clip_model (Module): Evaluating CLIP model

        Returns:
            Tensor: Labels
        """
        
        device =Device
        
        if inception_model == None:
            device = next(classifier.parameters()).device
            act_x = classifier.get_activation(x.to(device)).detach().cpu().data
            act_y = classifier.get_activation(y.to(device)).detach().cpu().data
            
            metric.update(act_x, act_y)
        else:
            act_x = inception_model(preprocess_inception(x).to(device))[0].detach().cpu().data.reshape(x.size(0), -1)
            act_y = inception_model(preprocess_inception(y).to(device))[0].detach().cpu().data.reshape(y.size(0), -1)
           
            metric.update(act_x, act_y)