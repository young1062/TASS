classdef (Abstract) prior
    properties 
        n_latent 
    end
    methods 
        function obj = prior(n_latent)
            obj.n_latent = n_latent;
        end
    end
end