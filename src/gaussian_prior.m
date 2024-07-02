classdef gaussian_prior < prior
    properties 
        mean
        var
    end
    
    methods 
        function obj = gaussian_prior(mean, var)
            obj@prior(length(mean));
            obj.mean = mean;
            obj.var = var;
        end
        
        function grad = gradient(obj, emit_param_gauss)
            grad = (obj.mean-emit_param_gauss.mu)./obj.var;
        end
    end
end