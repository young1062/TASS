function trans_param = random_transition_parameter(n_latent, pi_0) 
    A = rand(n_latent,n_latent);
    A_hat = rand(n_latent,n_latent);
    for i = 1:n_latent 
        A(:,i) = A(:,i)/sum(A(:,i)); 
        A_hat(:,i) = A_hat(:,i)/sum(A_hat(:,i)); 
    end 
    trans_param = transition_parameter(pi_0, A, A_hat);
end