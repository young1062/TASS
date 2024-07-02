classdef student_t_emission_parameter < emission_parameter
    properties 
        mu 
        sigmasq
        mu_grad 
        sigmasq_grad
        emit_dist
        nu
    end
    methods 
        function obj = student_t_emission_parameter(mu, sigmasq, nu, emit_dist)
            obj@emission_parameter(length(mu));
            obj.mu = mu;
            obj.sigmasq = sigmasq;
            obj.mu_grad = zeros(1,obj.n_latent);
            obj.sigmasq_grad = zeros(1,obj.n_latent);
            obj.emit_dist = emit_dist;
            obj.nu = nu;
        end
        
        function sim = emit_student_t(obj, z)
            sim = trnd(obj.nu)*sqrt(obj.sigmasq(z)) + obj.mu(z);
        end
        
        function dens = density_student_t(obj, yt) 
            dens = diag(tpdf((yt-obj.mu)./sqrt(obj.sigmasq),obj.nu)./sqrt(obj.sigmasq));
        end
        
       
        function sim = emit(obj, z)
            if obj.emit_dist == string("gaussian") 
                sim = obj.emit_gaussian(z);
            elseif obj.emit_dist == string("log_gaussian")
                sim = obj.emit_log_gaussian(z);
            elseif obj.emit_dist == string("student_t")
                sim = obj.emit_student_t(z);
            end
        end
        
        function P = calP(obj, yt) 
            if obj.emit_dist == string("gaussian")
                P = obj.density_gaussian(yt);
            elseif obj.emit_dist == string("log_gaussian")
                P = obj.density_log_gaussian(yt);
            elseif obj.emit_dist == string("student_t")
                P = obj.density_student_t(yt);
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
                        zt = (y(t)-obj.mu(j))/sqrt(obj.sigmasq(j));
                        rescaled_zt = (obj.nu+1)/obj.nu * zt / (1+zt^2/obj.nu);
                        Pyt = obj.calP(y(t));
                        [pipred,qpred] = approx_pred(y, t, sgmcmc_param, trans_param, obj);
                        Apipred = A*pipred;
                        grad_muj = grad_muj + Pyt(j,j)*qpred(j)*Apipred(j)*rescaled_zt/(qpred'*Pyt*Apipred*sqrt(obj.sigmasq(j)));
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
                        zt = (y(t)-obj.mu(j))/sqrt(obj.sigmasq(j));
                        rescaled_zt = (obj.nu+1)/obj.nu * zt / (1+1/obj.nu*zt^2);
                        Pyt = obj.calP(y(t));
                        [pipred,qpred] = approx_pred(y, t, sgmcmc_param, trans_param, obj);
                        Apipred = A*pipred;
                        grad_sigmasqj = grad_sigmasqj + 0.5*Pyt(j,j)*qpred(j)*Apipred(j)*(-1+zt*rescaled_zt)/... 
                                           (qpred'*Pyt*Apipred*obj.sigmasq(j));
                    end
                    obj.sigmasq_grad(j) = obj.sigmasq_grad(j) - grad_sigmasqj/w_sigmasq(i);
                end
            end
            obj.mu_grad = obj.mu_grad/mb_size;
            obj.sigmasq_grad = obj.sigmasq_grad/mb_size;
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