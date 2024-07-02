classdef (Abstract) emission_parameter
    properties 
        n_latent 
    end
    methods 
        function obj = emission_parameter(n_latent)
            obj.n_latent = n_latent;
        end
    end
end