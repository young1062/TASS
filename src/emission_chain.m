classdef (Abstract) emission_chain 
    properties 
        n_mcmc
        n_latent 
    end   
    methods 
        function obj = emission_chain(n_mcmc, n_latent)
            obj.n_mcmc = n_mcmc;
            obj.n_latent = n_latent;
        end
    end 
end