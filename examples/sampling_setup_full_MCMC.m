rng(1117676);
A = [.99 .005 .4950;.005 .99 .4950;.005 .005 .0100];
mu = [-20, 0, 20];
sigmasq = [1,1, 1];
pi0 = [1/3, 1/3, 1/3];
trans_param = transition_parameter(pi0, A, A);
emit_param = gaussian_emission_parameter(mu, sigmasq, "gaussian");
T = 1e5;
[z,y] = simHMM(trans_param, emit_param, T);
y_test = y((T/2+1):T);
y = y(1:T/2);
%%%%%%
%t_start = mod(length(y),15)+1;
%y = y(t_start:end);
n_latent = 3;
pi0_true = rand(1,n_latent); 
pi0_true = pi0_true/sum(pi0_true);
emit_dist = string("gaussian");
L = 2;
B = 5;
eps_clu = 3e-6;
eps_unif = 3e-6;
n_mcmc = 30000;
sgmcmc_param_clu = sgmcmc_parameter(L, B, eps_clu, n_mcmc);
sgmcmc_param_unif = sgmcmc_parameter(L, B, eps_unif, n_mcmc);
N = length(y)/(2*L+1);
%%%%Initial and prior 
trans_param = random_transition_parameter(n_latent, pi0_true);
%emit_param = student_t_emission_parameter([-1,0,1], ones(1,n_latent), nu, emit_dist);
emit_param = gaussian_emission_parameter([-1,0,1], ones(1,n_latent), emit_dist);
mb_size = 1;
mu_prior = gaussian_prior(zeros(1,n_latent), 10*ones(1,n_latent));
sigmasq_prior = IG_prior(3*ones(1,n_latent), 10*ones(1,n_latent));
emit_prior = gaussian_emission_prior(mu_prior, sigmasq_prior);
%%%%%Weights
clear wt_alex_clus wt_unif
eta = 1e-5;
[wt_alex_clus, center, xisum] = weights_alex_clus_leaveoneout(y, n_latent, N, L, eta);
wt_unif = weights_unif(n_latent, N);
A_prior = reshape(xisum, n_latent, n_latent);
prior_prob_mat = reshape(xisum,n_latent,n_latent);
prior_prob_mat = prior_prob_mat ./ sum(prior_prob_mat,1);
%trans_prior_info = dirichlet_prior(1e7*prior_prob_mat);
trans_prior_uninfo = dirichlet_prior(zeros([n_latent,n_latent])+1);
%%%%%%%
trans_param_init = random_transition_parameter(n_latent, pi0_true);
%emit_param_init = student_t_emission_parameter(zeros([1,n_latent]), 0.1*ones(1,n_latent),nu, emit_dist);
emit_param_init = gaussian_emission_parameter(zeros([1,n_latent]), 10*ones(1,n_latent), emit_dist);
clear trans_chain_alex_clus emit_chain_alex_clus trans_chain_alex_orac emit_chain_alex_orac 
clear trans_chain_unif emit_chain_unif
[trans_chain_alex_clus_uninfo, emit_chain_alex_clus_uninfo, A_hat_chain_clu, A_chain_clu, runtime_clu, runtimes_clu] = CSG_MCMC_IS(y, wt_alex_clus, mb_size, emit_param_init, ... 
                                                           trans_param_init, sgmcmc_param_clu,eps_clu, emit_prior, trans_prior_uninfo);
[trans_chain_unif, emit_chain_unif, A_hat_chain_unif, A_chain_unif, runtime_unif, runtimes_unif] = CSG_MCMC_IS(y, wt_unif, mb_size, emit_param_init, ... 
                                                      trans_param_init, sgmcmc_param_unif,eps_unif, emit_prior, trans_prior_uninfo);
eps_full = 1e-5;   
n_mcmc_full = 10000;
sgmcmc_param_full = sgmcmc_parameter(L, B, eps_full, n_mcmc_full);                                                
[trans_chain_full, emit_chain_full, A_hat_chain_clu, A_chain_clu, runtime_full, runtimes_full] = FULL_MCMC(y, wt_alex_clus, mb_size, emit_param_init, ... 
                                                           trans_param_init, sgmcmc_param_full, eps_full, emit_prior, trans_prior_uninfo);
save("full_sampling.mat")