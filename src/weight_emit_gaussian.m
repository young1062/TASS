classdef weight_emit_gaussian
    properties 
        mu
        sigmasq
    end
    methods 
        function obj = weight_emit_gaussian(mu, sigmasq)
            obj.mu = mu;
            obj.sigmasq = sigmasq;
        end
    end
end