## Standard libraries
import os
import numpy as np
import pdb

## Imports for plotting
import matplotlib.pyplot as plt
import matplotlib

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix


def log_sinkhorn(log_alpha, n_iter, gumbel_masks=None):
    """Performs incomplete Sinkhorn normalization to log_alpha.
    By a theorem by Sinkhorn and Knopp [1], a sufficiently well-behaved  matrix
    with positive entries can be turned into a doubly-stochastic matrix
    (i.e. its rows and columns add up to one) via the succesive row and column
    normalization.

    [1] Sinkhorn, Richard and Knopp, Paul.
    Concerning nonnegative matrices and doubly stochastic
    matrices. Pacific Journal of Mathematics, 1967
    Args:
      log_alpha: 2D tensor (a matrix of shape [N, N])
        or 3D tensor (a batch of matrices of shape = [batch_size, N, N])
      n_iters: number of sinkhorn iterations (in practice, as little as 20
        iterations are needed to achieve decent convergence for N~100)
    Returns:
      A 3D tensor of close-to-doubly-stochastic matrices (2D tensors are
        converted to 3D tensors with batch_size equals to 1)
    """
    inf = float('-inf')
    pad_max_len = log_alpha.shape[1]
    log_alpha = log_alpha.view(-1, pad_max_len, pad_max_len)

    for _ in range(n_iter):

        if gumbel_masks is not None:
            # Produces log_alpha's of size pad_max_len x pad_max_len where the K x K top-left submatrix is the actual permutation matrix, and the
            # rest are zeroes, where K is the actual sequence length.
            log_alpha = (log_alpha - (torch.logsumexp(log_alpha.masked_fill_(gumbel_masks, inf), dim=-1, keepdim=True)).view(-1, pad_max_len, 1))\
                .masked_fill_(gumbel_masks, inf)
            log_alpha = (log_alpha - (torch.logsumexp(log_alpha.masked_fill_(gumbel_masks, inf), dim=-2, keepdim=True)).view(-1, 1, pad_max_len))\
                .masked_fill_(gumbel_masks, inf)

        else:
            log_alpha = log_alpha - torch.logsumexp(log_alpha, -1, keepdim=True).view(-1, pad_max_len, 1)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, -2, keepdim=True).view(-1, 1, pad_max_len)

    return log_alpha.exp()

def matching(alpha):
    # Negate the probability matrix to serve as cost matrix. This function
    # yields two lists, the row and colum indices for all entries in the
    # permutation matrix we should set to 1.
    row, col = linear_sum_assignment(-alpha)

    # Create the permutation matrix.
    permutation_matrix = coo_matrix((np.ones_like(row), (row, col))).toarray()
    return torch.from_numpy(permutation_matrix)

def sample_gumbel(shape, device='cpu', eps=1e-20):
    """Samples arbitrary-shaped standard gumbel variables.
    Args:
      shape: list of integers
      eps: float, for numerical stability
    Returns:
      A sample of standard Gumbel random variables
    """
    u = torch.rand(shape, device=device)
    return -torch.log(-torch.log(u + eps) + eps)

def gumbel_sinkhorn(log_alpha, tau, n_iter, gumbel_masks=None):
    """ Sample a permutation matrix from the Gumbel-Sinkhorn distribution
    with parameters given by log_alpha and temperature tau.

    Args:
      log_alpha: Logarithm of assignment probabilities. In our case this is
        of dimensionality [num_pieces, num_pieces].
      tau: Temperature parameter, the lower the value for tau the more closely
        we follow a categorical sampling.
    """
    # Sample Gumbel noise.
    gumbel_noise = sample_gumbel(log_alpha.shape, device=log_alpha.device)

    # Apply the Sinkhorn operator!
    sampled_perm_mat = log_sinkhorn((log_alpha + gumbel_noise)/tau, n_iter,
        gumbel_masks=gumbel_masks)
    return sampled_perm_mat


if __name__ == "__main__":
  # Create a matrix containing random numbers.
  X = torch.rand((3, 3))
  S_X = log_sinkhorn(torch.log(X), n_iter=20)

  # Check whether rows and columns sum to 1.
  assert torch.allclose(S_X.sum(dim=0), torch.ones(S_X.shape[0]))
  assert torch.allclose(S_X.sum(dim=1), torch.ones(S_X.shape[1]))