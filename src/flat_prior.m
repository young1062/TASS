classdef flat_prior < prior
    properties 
        m
    end
    
    methods 
        function obj = flat_prior(m)
            obj@prior(length(m));
            obj.m = m;
        end
        
        function grad = gradient(obj, emit_param_gauss)
            grad = zeros(1,obj.n_latent);
        end
    end
end