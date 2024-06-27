# Sample Code for TASS-HMM

This is a repository containing a MATLAB implementation of the TASS-SGLD algorithm from
> Rihui Ou, Deborshee Sen, Alexander L Young, David B Dunson

> Targeted stochastic gradient Markoc chain Monte Carlo for hidden Markov models with rare latent states 

> arXiv:1810.13431

## Abstract

Markov chain Monte Carlo (MCMC) algorithms for hidden Markov models often rely on the forward-backward sampler. This makes them computationally slow as the length of the time series increases, motivating the recent development of sub-sampling-based approaches. These approximate the full posterior by using small random subsequences of the data at each MCMC iteration within stochastic gradient MCMC. In the presence of imbalanced data resulting from rare latent states, subsequences often exclude rare latent state data, leading to inaccurate inference and prediction/detection of rare events. We propose a targeted sub-sampling (TASS) approach that over-samples observations corresponding to rare latent states when calculating the stochastic gradient of parameters associated with them. TASS uses an initial clustering of the data to construct subsequence weights that reduce the variance in gradient estimation. This leads to improved sampling efficiency, in particular in settings where the rare latent states correspond to extreme observations. We demonstrate substantial gains in predictive and inferential accuracy on real and synthetic examples. 

## Getting Started

This repository is publically available and may be cloned from GitHub

```bash
git clone git@github.com:young1062/TASS.git
```



