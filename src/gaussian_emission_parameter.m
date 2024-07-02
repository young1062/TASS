classdef gaussian_emission_parameter < emission_parameter
    properties 
        mu 
        sigmasq
        mu_grad 
        sigmasq_grad
        emit_dist
    end
    methods 
        function obj = gaussian_emission_parameter(mu, sigmasq, emit_dist)
            obj@emission_parameter(length(mu));
            obj.mu = mu;
            obj.sigmasq = sigmasq;
            obj.mu_grad = zeros(1,obj.n_latent);
            obj.sigmasq_grad = zeros(1,obj.n_latent);
            obj.emit_dist = emit_dist;
        end
        
        function sim = emit_gaussian(obj, z)
            sim = normrnd(obj.mu(z), sqrt(obj.sigmasq(z)));
        end
        
        function sim = emit_log_gaussian(obj, z)
            sim = lognrnd(obj.mu(z), sqrt(obj.sigmasq(z)));
        end
        
        function dens = density_gaussian(obj, yt) 
            dens = diag(normpdf(yt, obj.mu, sqrt(obj.sigmasq)));
        end
        
        function dens = density_log_gaussian(obj, yt) 
            dens = diag(lognpdf(yt, obj.mu, sqrt(obj.sigmasq)));
        end
        
        function sim = emit(obj, z)
            if obj.emit_dist == string("gaussian") 
                sim = obj.emit_gaussian(z);
            elseif obj.emit_dist == string("log_gaussian")
                sim = obj.emit_log_gaussian(z);
            end
        end
        
        function P = calP(obj, yt) 
            if obj.emit_dist == string("gaussian")
                P = obj.density_gaussian(yt);
            elseif obj.emit_dist == string("log_gaussian")
                P = obj.density_log_gaussian(yt);
            end
        end
        
        
        function obj = eval_stoch_grad(obj, y, b, clusters, trans_param, sgmcmc_param, prior)
            L = sgmcmc_param.L;
            n_latent = trans_param.n_latent;
            A = trans_param.A;
            obj.mu_grad = prior.mu_prior.gradient(obj); 
            obj.sigmasq_grad = prior.sigmasq_prior.gradient(obj);
            for j = 1:n_latent 
                n = clusters.emit{j}.counts;
                minibatch = form_mb(clusters.emit{j}, b);
                for i = 1:minibatch.n_clusters
                    grad_muj = 0;
                    grad_sigmasqj = 0;
                    mbi = cell2mat(minibatch.mb(i));
                    for s = 1:length(mbi)
                        tau = mbi(s);
                        for t = (tau-L):(tau+L) 
                            zt = y(t) - obj.mu(j);
                            Pyt = obj.calP(y(t));
                            [pipred,qpred] = approx_pred(y, t, sgmcmc_param, trans_param, obj);
                            Apipred = A*pipred;
                            grad_muj = grad_muj + Pyt(j,j)*qpred(j)*Apipred(j)*zt/(qpred'*Pyt*Apipred*obj.sigmasq(j));
                            grad_sigmasqj = grad_sigmasqj + 0.5*Pyt(j,j)*qpred(j)*Apipred(j)*(-obj.sigmasq(j)+zt^2)/... 
                                               (qpred'*Pyt*Apipred*obj.sigmasq(j)^2);
                                               
                        end
                    end
                    obj.mu_grad(j) = obj.mu_grad(j) - grad_muj*n(i)/b(i);
                    obj.sigmasq_grad(j) = obj.sigmasq_grad(j) - grad_sigmasqj*n(i)/b(i);
                end
            end
        end
        
        function obj = eval_stoch_grad_IS(obj, y, weights, mb_size, trans_param, sgmcmc_param, prior)
            L = sgmcmc_param.L;
            ll = 2*L+1;
            n_latent = trans_param.n_latent;
            A = trans_param.A;
            obj.mu_grad = prior.mu_prior.gradient(obj); 
            obj.sigmasq_grad = prior.sigmasq_prior.gradient(obj);
            for j = 1:n_latent 
                w_mu = weights.emit.mu{j};
                mb_mu = randsample(length(w_mu),mb_size,true,w_mu);
                for s = 1:length(mb_mu)
                    grad_muj = 0;
                    i = mb_mu(s);
                    tau = (2*L+1)*i-L;
                    for t = ((i-1)*ll+1):(i*ll)
                        zt = y(t) - obj.mu(j);
                        Pyt = obj.calP(y(t));
                        [pipred,qpred] = approx_pred(y, t, sgmcmc_param, trans_param, obj);
                        Apipred = A*pipred;
                        grad_muj = grad_muj + Pyt(j,j)*qpred(j)*Apipred(j)*zt/(qpred'*Pyt*Apipred*obj.sigmasq(j));
                    end
                    obj.mu_grad(j) = obj.mu_grad(j) - grad_muj/w_mu(i);
                end
                w_sigmasq = weights.emit.sigmasq{j};
                mb_sigmasq = randsample(length(w_sigmasq),mb_size,true,w_sigmasq);
                for s = 1:length(mb_sigmasq)
                    grad_sigmasqj = 0;
                    i = mb_sigmasq(s);
                    tau = (2*L+1)*i-L;
                    for t = ((i-1)*ll+1):(i*ll)
                        zt = y(t) - obj.mu(j);
                        Pyt = obj.calP(y(t));
                        [pipred,qpred] = approx_pred(y, t, sgmcmc_param, trans_param, obj);
                        Apipred = A*pipred;
                        grad_sigmasqj = grad_sigmasqj + 0.5*Pyt(j,j)*qpred(j)*Apipred(j)*(-obj.sigmasq(j)+zt^2)/... 
                                           (qpred'*Pyt*Apipred*obj.sigmasq(j)^2);
                    end
                    obj.sigmasq_grad(j) = obj.sigmasq_grad(j) - grad_sigmasqj/w_sigmasq(i);
                end
            end
            obj.mu_grad = obj.mu_grad/mb_size;
            obj.sigmasq_grad = obj.sigmasq_grad/mb_size;
        end
        
        function obj = eval_full_grad(obj, y, trans_param, sgmcmc_param, prior)
            T = length(y);
            n_latent = trans_param.n_latent;
            A = trans_param.A;
            obj.mu_grad = prior.mu_prior.gradient(obj); 
            obj.sigmasq_grad = prior.sigmasq_prior.gradient(obj);
            for j = 1:n_latent 
                %%%%%mu
                grad_muj = 0;
                for t = 1:T
                    zt = y(t) - obj.mu(j);
                    Pyt = obj.calP(y(t));
                    [pipred, qpred] = approx_pred(y, t, sgmcmc_param, trans_param, obj);
                    Apipred = A*pipred;
                    grad_muj = grad_muj + Pyt(j,j)*qpred(j)*Apipred(j)*zt/(qpred'*Pyt*Apipred*obj.sigmasq(j));
                end
                obj.mu_grad(j) = obj.mu_grad(j) - grad_muj;
                %%%%%sigmasq
                grad_sigmasqj = 0;
                for i = 1:T
                     zt = y(t) - obj.mu(j);
                     Pyt = obj.calP(y(t));
                     [pipred,qpred] = approx_pred(y, t, sgmcmc_param, trans_param, obj);
                     Apipred = A*pipred;
                     grad_sigmasqj = grad_sigmasqj + 0.5*Pyt(j,j)*qpred(j)*Apipred(j)*(-obj.sigmasq(j)+zt^2)/... 
                                           (qpred'*Pyt*Apipred*obj.sigmasq(j)^2);
                end
                obj.sigmasq_grad(j) = obj.sigmasq_grad(j) - grad_sigmasqj;
                %%%%%      
            end
        end
        
        function obj = SGLD_update(obj, sgmcmc_param)
            eps = sgmcmc_param.eps;
            oldsigmasq = obj.sigmasq;
            for j = 1:obj.n_latent
                sd = sqrt(eps*(2*obj.sigmasq(j)));
                obj.mu(j) = obj.mu(j) - eps*obj.sigmasq(j)*obj.mu_grad(j) + sd*randn();
                obj.sigmasq(j) = obj.sigmasq(j) - eps*(obj.sigmasq(j)^2*obj.sigmasq_grad(j) + obj.sigmasq(j)) + sd*randn();
            end
            % Reject sigmasq samples that are not positive 
            %%% WHAT? %%%
            if sum(obj.sigmasq>0) < obj.n_latent
                obj.sigmasq = oldsigmasq;
            end
            obj.mu_grad = zeros(1,obj.n_latent);
            obj.sigmasq_grad = zeros(1,obj.n_latent);
        end
        
    end
end