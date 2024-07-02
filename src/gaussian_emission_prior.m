classdef gaussian_emission_prior
    properties 
        mu_prior
        sigmasq_prior
    end
    methods 
        function obj = gaussian_emission_prior(mu_prior, sigmasq_prior)
            obj.mu_prior = mu_prior;
            obj.sigmasq_prior = sigmasq_prior;
        end
    end
end