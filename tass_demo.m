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
