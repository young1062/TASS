function [z,y] = sim_HMM(trans_param, emit_param, T)
    z = zeros(T,1);
    y = zeros(T,1);
    K = length(trans_param.pi_0);
    z(1) = randsample(K, 1, true, trans_param.pi_0);

    for t = 2:T
        z(t) = randsample(K, 1, true, trans_param.A(:,z(t-1)));
    end

    for t = 1:T
        %Simulate the observation series y from z
        y(t) = emit_param.emit(z(t));
    end