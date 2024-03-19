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




def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            covmean = np.nan;
            print('Imaginary component {}'.format(m));
            # raise ValueError('Imaginary component {}'.format(m))
        else:
            covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


class FrechetDistance(Metric):
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

    def __init__(self, feature: int = 512, limit: int = 30000,
                 **kwargs: Dict[str, Any]) -> None:
        super().__init__(**kwargs)

        self.limit = limit
        self._debug = True
        self._dtype = torch.float64

        for k in ['x', 'y']:  # x: real, y: text, x0: fake
            self.add_state(f"{k}_feat", [], dist_reduce_fx=None)

    def update(self, x: Tensor, y: Tensor) -> None:
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
        assert x.shape[0] == y.shape[0] and x.shape[1] == y.shape[1]

        self.orig_dtype = x.dtype
        x, y = [x for x in [x, y]]
        self.x_feat.append(x)
        self.y_feat.append(y)


    def compute(self, reduction: bool = True, mode=None) -> Tensor:
        r"""
        Calculate the CLIP-S score based on accumulated extracted features.
        """
        feats = [torch.cat(getattr(self, f"{k}_feat"), dim=0)
                 for k in ['x', 'y']]

        return self._compute(*feats, reduction)

    def _compute(self, X: Tensor, Y: Tensor,  reduction):
       
        excess = X.shape[0] - self.limit
        if 0 < excess:
            X, Y = [x[:-excess] for x in [X, Y]]
        print(X.shape)
        print(Y.shape)
        x_mu , x_sgima = calculate_activation_statistics(X)
        y_mu, y_sigma = calculate_activation_statistics(Y)
        return calculate_frechet_distance(mu1=x_mu,sigma1=x_sgima,mu2=y_mu,sigma2= y_sigma )



def calculate_activation_statistics(act):
    """Calculation of the statistics used by the FID.
    Params:
    -- act: 
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """

    mu = np.mean(act.cpu().numpy(), axis=0)
    print("mean computed")
    sigma = np.cov(act.cpu().numpy(), rowvar=False)
    print("sigma computed")
    return mu, sigma





