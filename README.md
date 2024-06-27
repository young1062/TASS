# Sample Code for TASS-HMM

This is a repository containing a MATLAB implementation of the TASS-SGLD algorithm from
> Rihui Ou, Deborshee Sen, Alexander L Young, David B Dunson
> "Targeted stochastic gradient Markoc chain Monte Carlo for hidden Markov models with rare latent states" 
> arXiv:1810.13431

## Abstract

Markov chain Monte Carlo (MCMC) algorithms for hidden Markov models often rely on the forward-backward sampler. This makes them computationally slow as the length of the time series increases, motivating the recent development of sub-sampling-based approaches. These approximate the full posterior by using small random subsequences of the data at each MCMC iteration within stochastic gradient MCMC. In the presence of imbalanced data resulting from rare latent states, subsequences often exclude rare latent state data, leading to inaccurate inference and prediction/detection of rare events. We propose a targeted sub-sampling (TASS) approach that over-samples observations corresponding to rare latent states when calculating the stochastic gradient of parameters associated with them. TASS uses an initial clustering of the data to construct subsequence weights that reduce the variance in gradient estimation. This leads to improved sampling efficiency, in particular in settings where the rare latent states correspond to extreme observations. We demonstrate substantial gains in predictive and inferential accuracy on real and synthetic examples. 

## Getting Started

This repository is publicly available and may be cloned from GitHub

```bash
git clone git@github.com:young1062/TASS.git
```

## Basic Demo

The full TASS-HMM algorithm is implemented in the function CSG_MCMC_IS and may be called in MATLAB as follows:

```matlab
[trans_chain_alex_clus_uninfo, emit_chain_alex_clus_uninfo, A_hat_chain_clu, A_chain_clu, runtime_clu, runtimes_clu] = CSG_MCMC_IS(time_series, weights, mb_size, emit_param_init, trans_param_init, sgmcmc_param_full, eps_full, emit_prior, trans_prior_uninfo)
```

#### Inputs 
| Variable | Explanation |
|--------|-------------|
| time_series | |
| weights | |
| mb_size | |
| emit_param_init | |
| trans_param_init | |
| sgmcmc_param_full | |
| eps_full |  |
| emit_prior | |
| trans_prior_uninfo | |

#### Outputs

| Variable | Explanation |
| trans_chain |  |
| emit_chain |   |
| Ahat_chain |   |
| Achain_clu |   |
|runtime_clu | runtimes_clu |

## Data

## Sample results from paper

## Having issues?
If you have any troubles please file the issue through the repository page on github.com.




