classdef gaussian_emission_chain < emission_chain
    properties 
        mu
        sigmasq
    end
    methods 
        function obj = gaussian_emission_chain(n_mcmc, n_latent)
           obj@emission_chain(n_mcmc+1, n_latent);
           obj.mu = zeros(n_mcmc+1, n_latent);
           obj.sigmasq = zeros(n_mcmc+1, n_latent);
        end
        function obj = save(obj, gauss_emit_param, itr) 
            obj.mu(itr+1,:) = gauss_emit_param.mu;
            obj.sigmasq(itr+1,:) = gauss_emit_param.sigmasq;
        end
    end
end