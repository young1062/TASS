classdef IG_prior < prior
    properties 
        a
        b
    end
    
    methods 
        function obj = IG_prior(a, b)
            obj@prior(length(a));
            obj.a = a;
            obj.b = b;
        end
        
        function grad = gradient(obj, emit_param_gauss)
            grad = (obj.a+1)./emit_param_gauss.sigmasq - obj.b./emit_param_gauss.sigmasq.^2;
        end
    end
end
  