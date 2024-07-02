function [pipred,qpred] = approx_pred(y, t, sgmcmc_param, trans_param, emit_param)
    T = size(y,1);
    n_latent = emit_param.n_latent;
    A = trans_param.A;
    B = sgmcmc_param.B;
    
    qpred = eye(n_latent);
    pipred = trans_param.pi_0';
    if t < (B+1)
        for i = 1:(t-1)
            pipred = emit_param.calP(y(i))*A*pipred;
        end
        for i = (t+1):(t+B)
            qpred = emit_param.calP(y(i))*A*qpred;
        end
    elseif t > (T-B-1)
        for i = (t-B):(t-1)
            pipred = emit_param.calP(y(i))*A*pipred;
        end
        for i = (t+1):T
            qpred = emit_param.calP(y(i))*A*qpred;
        end
    else
        for i = (t-B):(t-1)
            pipred = emit_param.calP(y(i))*A*pipred;
        end
        for i = (t+1):(t+B)
            qpred = emit_param.calP(y(i))*A*qpred;
        end
    end
    qpred = ones(n_latent,1)'*qpred;
    qpred = qpred';
end