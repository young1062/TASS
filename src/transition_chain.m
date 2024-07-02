classdef transition_chain
    properties 
        A
        n_mcmc
        n_latent
    end
    methods 
        function obj = transition_chain(n_mcmc, n_latent) 
            obj.A = zeros(n_mcmc+1, n_latent, n_latent);
            obj.n_mcmc = n_mcmc;
            obj.n_latent = n_latent;
        end
        function obj = save(obj, trans_param, itr)
            obj.A(itr+1,:,:) = trans_param.A;
        end
    end
end