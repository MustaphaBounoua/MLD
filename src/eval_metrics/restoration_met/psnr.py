"""
    mid.metric 
    Copyright (c) 2022-present NAVER Corp.
    Apache-2.0
"""

import torch




"""
    mid.metric 
    Copyright (c) 2022-present NAVER Corp.
    Apache-2.0
"""
from typing import *
import random

import torch
from torch import Tensor
from scipy import linalg
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_info
import numpy as np
from torchmetrics.image import PeakSignalNoiseRatio




class PSNR(Metric):
    r"""
    Calculates FrechetDistance between two dist

    Args:
        feature (int): the number of features
        limit (int): limit the number of samples to calculate
        kwargs:
            Additional keyword arguments, see :ref:`Metric kwargs` for more info.
    """
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self, limit: int = 30000,
                 **kwargs: Dict[str, Any]) -> None:
        super().__init__(**kwargs)

        self.limit = limit
        self._debug = True
        self._dtype = torch.float64

        for k in ['x']:  # x: real, y: text, x0: fake
            self.add_state(f"{k}_feat", [], dist_reduce_fx=None)


    def update(self, x: Tensor) -> None:
        r"""
        Update the state with extracted features in double precision. This 
        method changes the precision of features into double-precision before 
        saving the features. 
X_ref, Y_ref, X
        Args:
            x (Tensor): tensor with the extracted real image features
            y (Tensor): tensor with the extracted text features
            x0 (Tensor): tensor with the extracted fake image features
        """
        
        self.x_feat.append(float(x))
      


    def compute(self, reduction: bool = True, mode=None) -> Tensor:
        r"""
        Calculate the CLIP-S score based on accumulated extracted features.
        """
        feats = getattr(self, f"x_feat")
        #feats = [torch.cat(getattr(self, f"{k}_feat"), dim=0)
         #        for k in ['x']]
        return self._compute(feats)


    def _compute(self, X: Tensor):
       
        return torch.tensor([np.mean(X)]).numpy()[0]







def init_metric_psnr(limit = 30000) :
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

    m = PSNR(limit=limit)
   # m.cuda(device)
    m._debug = False
    return m


@torch.no_grad()
def populate_metrics_psnr(metric, x, y , classifier=None, inception_model=None,device=None):
        """Populate the list of metrics using a given data loader.

        Args:
            dataloader (DataLoader): Data loader
            metrics (List[Metric]): List of metrics
            clip_model (Module): Evaluating CLIP model

        Returns:
            Tensor: Labels
        """
        Psnr = PeakSignalNoiseRatio()
        psnr = Psnr(x.detach().cpu(), y.detach().cpu()).numpy()
      
        metric.update(psnr)


