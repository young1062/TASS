# Sample Code for TASS-HMM

This is a repository containing a MATLAB implementation of the TASS-HMM algorithm from
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

The full TASS-HMM algorithm is implemented in the function CSG_MCMC_IS and may be called in MATLAB as follows:

```matlab
[trans_chain, emit_chain, A_hat_chain, A_chain, runtime, runtimes] = CSG_MCMC_IS(time_series, weights, mb_size, ...
                                          emit_param_init, trans_param_init, ... 
					  sgmcmc_param, emit_prior, trans_prior);
```

At present only Gaussian emissions are supported.

#### Inputs 
| Variable | Explanation |
|--------|-------------|
| time_series | Sequential observations of the time series|
| weights | Object of class weights giving assigned weights to each subseries by component|
| mb_size | Size of the mini batch (1 by default) |
| emit_param_init | Object of class gaussian_emission_parameter specifying the initial values of the emission distribution |
| trans_param_init | Object of class random_transition_parameter specifying the initial value of the transition matrix |
| sgmcmc_param | Object of class sgmcmc_parameter containing specifying the buffer size (B), sequence length (L), step size (eps), and number of steps (n_mcmc)|
| emit_prior | Object of class Gaussian emission prior specifying prior on the Gaussian emissions |
| trans_prior | Object of class dirichlet prior specifying the prior(s) for the transition matrix|

#### Outputs

| Variable | Explanation |
|--------|-------------|
| trans_chain | Posterior samples of hidden state |
| emit_chain | Posterior samples of emission parameters  |
| A_hat_chain | Posterior sample of transition matrix after normalization |
| A_chain |  Posterior samples of Transition matrix prior to normalization|
| runtime |  Total runtime |
| runtimes | Run time recorded at 10% of samples | 

## Basic Demo

```matlab
%% Specify parameters and generate a realization of an HMM
rng(1) % for reproducibility
% transition matrix
A = [.99 .005 .4950;.005 .99 .4950;.005 .005 .0100];
% centers of Gaussian emissions
mu = [-20, 0, 20];
% variances of Gaussian emissions
sigmasq = [1,1, 1];
% specifying prior for initial hidden state
pi0 = [1/3, 1/3, 1/3];
% store transition parameters to object of class transition_parameter
trans_param = transition_parameter(pi0, A, A);
% store emission parameters to object of class gaussian_emission_parameter
emit_param = gaussian_emission_parameter(mu, sigmasq, "gaussian");
% specify length of time series
T = 1e3;
% generate a sample from the HMM
[z,y] = simHMM(trans_param, emit_param, T);

%% Set and run posterior sampler
% specify minibatch parameters and 
L = 2; % half sequence length 
B = 5; % buffer size
eps = 3e-6; % step size
n_mcmc = 1000; % total number of MCMC steps
mb_size = 1;
% store SGMCM in object of class sgmcmc_parameter
sgmcmc_param = sgmcmc_parameter(L, B, eps, n_mcmc);

% specify emissions and associated priors
n_latent = 3; % specify number of hidden states
emit_dist = string("gaussian"); % specify gaussian emissions
emit_param = gaussian_emission_parameter([-1,0,1], ones(1,n_latent), emit_dist);
mu_prior = gaussian_prior(zeros(1,n_latent), 10*ones(1,n_latent));
sigmasq_prior = IG_prior(3*ones(1,n_latent), 10*ones(1,n_latent));
emit_prior = gaussian_emission_prior(mu_prior, sigmasq_prior);

% specify transition prior
trans_param = random_transition_parameter(n_latent, [1,1,1]);
trans_prior = dirichlet_prior(zeros([n_latent,n_latent])+1);



%% Compute sampling weights for minibatch 
N = length(y)/(2*L+1); % number of subsequences
%%%%%Weights
clear wt_alex_clus wt_unif
eta = 1e-5;
[wt, center, xisum] = weights_by_clustering(y, n_latent, N, L, eta);
A_prior = reshape(xisum, n_latent, n_latent);
prior_prob_mat = reshape(xisum,n_latent,n_latent);
prior_prob_mat = prior_prob_mat ./ sum(prior_prob_mat,1);

%% run TASS
% specify initial conditions
trans_param_init = random_transition_parameter(3,[1,1,1]);
emit_param_init = gaussian_emission_parameter(zeros([1,n_latent]),... % means
    100*zeros([1,n_latent]), ... % variances
    'gaussian');

[trans_chain, emit_chain, A_hat, A, runtime_clu, runtimes_clu] = CSG_MCMC_IS(y, wt, mb_size, emit_param_init, ... 
                                                           trans_param_init, sgmcmc_param, emit_prior, trans_prior);
```


## Sample results from paper

The script used to generate the results comparing the weighted subsampling TASS-HMM against Uniform Sub Sampling and Full Sampling (shown in Section 4 of the manuscript) is available in examples/paper.m


## Having issues?
If you have any troubles please file the issue through the repository page on github.com.




