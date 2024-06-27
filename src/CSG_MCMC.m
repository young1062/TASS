function [trans_chain, emit_chain] = CSG_MCMC(y, clusters, b, emit_param, trans_param, sgmcmc_param, prior)

    emit_chain = gaussian_emission_chain(sgmcmc_param.n_mcmc, emit_param.n_latent);
    trans_chain = transition_chain(sgmcmc_param.n_mcmc, emit_param.n_latent);
    trans_chain = trans_chain.save(trans_param, 0);
    emit_chain = emit_chain.save(emit_param, 0);
    
    start = cputime;
    for itr = 1:sgmcmc_param.n_mcmc
        clear minibatch trans_grad emit_grad;

        trans_param = trans_param.eval_stoch_grad(y, b, clusters, emit_param, sgmcmc_param);
        trans_param = trans_param.SGLD_update(sgmcmc_param);

        emit_param = emit_param.eval_stoch_grad(y, b, clusters, trans_param, sgmcmc_param, prior);
        emit_param = emit_param.SGLD_update(sgmcmc_param);

        % Save parameters in each iteration
        trans_chain = trans_chain.save(trans_param, itr);
        emit_chain = emit_chain.save(emit_param, itr);
        
        if rem(itr,sgmcmc_param.n_mcmc/10) == 0 
            disp(10*itr/(sgmcmc_param.n_mcmc/10)+"% MCMC iterations in "+(cputime-start)/60+" mins");
        end
    end
end