function [trans_chain, emit_chain, A_hat_chain, A_chain, runtime, runtimes] = CSG_MCMC_IS(y, weights, mb_size, emit_param, trans_param, sgmcmc_param, trans_eps, emit_prior, trans_prior)
    sgmcmc_param_trans = sgmcmc_param;
    sgmcmc_param_trans.eps = trans_eps;
    emit_chain = gaussian_emission_chain(sgmcmc_param.n_mcmc, emit_param.n_latent);
    trans_chain = transition_chain(sgmcmc_param.n_mcmc, emit_param.n_latent);
    trans_chain = trans_chain.save(trans_param, 0);
    emit_chain = emit_chain.save(emit_param, 0);
    A_hat_chain = zeros(emit_param.n_latent, emit_param.n_latent, sgmcmc_param.n_mcmc);
    A_chain = zeros(emit_param.n_latent, emit_param.n_latent, sgmcmc_param.n_mcmc);
    start = cputime;
    runtimes = zeros([10,1]);
    count = 1;
    for itr = 1:sgmcmc_param.n_mcmc
        clear minibatch trans_grad emit_grad;

        trans_param = trans_param.eval_stoch_grad_IS(y, weights, mb_size, emit_param, sgmcmc_param_trans, trans_prior);
        trans_param = trans_param.SGLD_update(sgmcmc_param_trans);

        emit_param = emit_param.eval_stoch_grad_IS(y, weights, mb_size, trans_param, sgmcmc_param, emit_prior);
        emit_param = emit_param.SGLD_update(sgmcmc_param);

        % Save parameters in each iteration
        trans_chain = trans_chain.save(trans_param, itr);
        emit_chain = emit_chain.save(emit_param, itr);
        
        if rem(itr,sgmcmc_param.n_mcmc/10) == 0 
            disp(10*itr/(sgmcmc_param.n_mcmc/10)+"% MCMC iterations in "+(cputime-start)/60+" mins");
            runtimes(count) = cputime-start;
            count = count + 1;
        end
        
        %%%%Renormalize A_hat
        if (mod(itr,10)==0)
            trans_param.A_hat = abs(trans_param.A_hat);
            trans_param.A_hat = trans_param.A_hat ./ sum(trans_param.A_hat,1);
        end
       
        %%%Save A_hat
        A_hat_chain(:,:,itr) = trans_param.A_hat;
        A_chain(:,:,itr) = trans_param.A;
    end
    runtime = cputime - start;
end