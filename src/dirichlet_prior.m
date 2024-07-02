classdef dirichlet_prior < prior
    properties 
        alpha
    end
    
    methods 
        function obj = dirichlet_prior(alpha)
            obj@prior(length(alpha));
            obj.alpha = alpha;
        end
        
        function grad = gradient(obj, trans_param)
            grad = zeros(obj.n_latent,obj.n_latent);
            for i = 1:obj.n_latent 
                grad(:,i) = (obj.alpha(:,i)-1)./trans_param.A_hat(:,i);
            end
        end
    end
end