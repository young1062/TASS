classdef sgmcmc_parameter
    properties 
        L
        B
        eps
        n_mcmc
    end
    methods 
        function obj = sgmcmc_parameter(L, B, eps, n_mcmc) 
            obj.L = L;
            obj.B = B;
            obj.eps = eps;
            obj.n_mcmc = n_mcmc;
        end
    end
end