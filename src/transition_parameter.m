classdef transition_parameter
    properties
        pi_0
        A
        A_hat
        A_grad
    end
    properties (Dependent) 
        n_latent; 
    end
    methods 
        function obj = transition_parameter(pi_0, A, A_hat) 
            obj.pi_0 = pi_0;
            obj.A = A;
            obj.A_hat = A_hat;
            obj.A_grad = zeros(size(A));
        end
        
        function obj = random_transion_parameter(n_latent, pi_0) 
            obj.A = rand(n_latent,n_latent);
            obj.A_hat = rand(n_latent,n_latent);
            for i = 1:n_latent 
                obj.A(:,i) = obj.A(:,i)/sum(obj.A(:,i)); 
                obj.A_hat(:,i) = obj.A_hat(:,i)/sum(obj.A_hat(:,i)); 
            end 
        end
        
        function n_latent = get.n_latent(obj)
            n_latent = length(obj.pi_0);
        end
        
        function obj = eval_stoch_grad(obj, y, b, clusters, emit_param, sgmcmc_param, prior)
            L = sgmcmc_param.L;
            n_latent = obj.n_latent;
            obj.A_grad = prior.gradient(obj);
            for j = 1:n_latent
                for k = 1:n_latent
                    n = clusters.trans{j,k}.counts;
                    minibatch = form_mb(clusters.trans{j,k}, b);
                    for i = 1:minibatch.n_clusters
                        grad_A = 0;
                        mbi = cell2mat(minibatch.mb(i));
                        for s = 1:length(mbi)
                            tau = mbi(s);
                            for t = (tau-L):(tau+L)
                                Pyt = emit_param.calP(y(t));
                                [pipred, qpred] = approx_pred(y, t, sgmcmc_param, obj, emit_param);
                                grad_A = grad_A + Pyt(j,j)*qpred(j)*pipred(k)./(qpred'*Pyt*obj.A*pipred);
                            end
                        end
                        obj.A_grad(j,k) = obj.A_grad(j,k) - grad_A*n(i)/b(i);
                    end
                end
            end
        end
        
        function obj = eval_stoch_grad_IS(obj, y, weights, mb_size, emit_param, sgmcmc_param, prior)
            L = sgmcmc_param.L;
            ll = 2*L+1;
            n_latent = obj.n_latent;
            obj.A_grad = prior.gradient(obj);
            for j = 1:n_latent
                for k = 1:n_latent
                    w = weights.trans{j,k};
                    mb = randsample(length(w),mb_size,true,w);
                    for s = 1:length(mb)
                        grad_A = 0;
                        i = mb(s);
                        tau = ll*i-L;
                        %for t = (tau-L):(tau+L)
                        for t = ((i-1)*ll+1):(i*ll)
                            Pyt = emit_param.calP(y(t));
                            [pipred, qpred] = approx_pred(y, t, sgmcmc_param, obj, emit_param);
                            grad_A = grad_A + Pyt(j,j)*qpred(j)*pipred(k)./(qpred'*Pyt*obj.A*pipred);
                        end
                        obj.A_grad(j,k) = obj.A_grad(j,k) - grad_A/w(i);
                    end
                end
            end
            obj.A_grad = obj.A_grad/mb_size;
        end
        
        function obj = eval_grad_full(obj,y, emit_param, sgmcmc_param,  prior)
            T = length(y);
            obj.A_grad = prior.gradient(obj);
            n_latent = obj.n_latent;
            for j = 1:n_latent
                for k = 1:n_latent
                    grad_A = 0;
                    for t = 1:T
                        Pyt = emit_param.calP(y(t));
                        [pipred, qpred] = approx_pred(y, t, sgmcmc_param, obj, emit_param);
                        grad_A = grad_A + Pyt(j,j)*qpred(j)*pipred(k)./(qpred'*Pyt*obj.A*pipred);
                    end
                    obj.A_grad(j,k) = obj.A_grad(j,k) - grad_A;
                end
            end             
        end
        
        function obj = SGLD_update(obj, sgmcmc_param)
            eps = sgmcmc_param.eps;
            n_latent = obj.n_latent;
            obj.A_hat = abs(obj.A_hat - eps*(obj.A_hat.*obj.A_grad+1) + sqrt(eps*(2*obj.A_hat)).*normrnd(0,1,n_latent,n_latent));
            obj.A = abs(obj.A_hat)./sum(abs(obj.A_hat)); 
            obj.A_grad = zeros(size(obj.A));
        end
    end
end


