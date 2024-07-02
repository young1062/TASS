function [weights_alex, center, xisum] = weights_alex_clus_leaveoneout(y, n_latent, N, L, eta) 

    [z, center] = kmeans(y, n_latent, 'MaxIter', 1000); %Kmeans with K clusters in total
    
    ll = 2*L+1;
    clear ybar c ybar_chunk s2 s2_chunk
    for j = 1:n_latent 
        ybar(j) = mean(y(z==j));
        s2(j) = var(y(z==j));
        for n = 1:N 
            idx = ((n-1)*ll+1):(n*ll);
            c(n,j) = sum(z(idx)==j);
            if c(n,j) == 0 
                ybar_chunk(n,j) = 0;
                s2_chunk(n,j) = 0;
            else 
                ybar_chunk(n,j) = mean(y(z(idx)==j));
                s2_chunk(n,j) = sum((y(idx)-ybar(j)).^2)/c(n,j);
            end
        end
    end
    
    clear weights_mu weights_sigmasq
    for j = 1:n_latent 
        for n = 1:N 
           weights_mu(n,j) = c(n,j)*abs(ybar_chunk(n,j)-ybar(j)) + eta^2;
           weights_sigmasq(n,j) = c(n,j)*(s2(j)+s2_chunk(n,j)) + eta^2;
        end
        weights_mu(:,j) = weights_mu(:,j)/sum(weights_mu(:,j));
        weights_sigmasq(:,j) = weights_sigmasq(:,j)/sum(weights_sigmasq(:,j));
    end
    
    clear xi weight_A;
    clen = 2*L+1;
    for i = 1:n_latent 
        for j = 1:n_latent 
            for n = 1:N
                idx = ((n-1)*ll+1):(n*ll);
                xi(n,i,j) = sum((z(idx(1:(clen-1)))==i).*(z(idx(2:clen))==j));
            end
        end
    end
    
    xisum = sum(xi,1) + 0.5;
    A_map_est = reshape(xisum,n_latent,n_latent);
    A_map_est = A_map_est ./ sum(A_map_est,1);
    %%%
    for n = 1:N 
        for i = 1:n_latent 
            for j = 1:n_latent
                weight_A(n,i,j) =  xi(n,i,j) ./ A_map_est(j,i) + 1e-10;
            end
        end
    end
   
    %%%%
    for i = 1:n_latent 
        for j = 1:n_latent
          weight_A(:,i,j) = weight_A(:,i,j)/sum(weight_A(:,i,j));
        end
    end
    
    clear wt_mu wt_sigmasq
    for j = 1:n_latent 
        wt_mu{j} = weights_mu(:,j);
        wt_sigmasq{j} = weights_sigmasq(:,j);
    end

    weights_emit = weight_emit_gaussian(wt_mu, wt_sigmasq);

    clear weights_trans
    for i = 1:n_latent 
        for j = 1:n_latent
            weights_trans{j,i} = weight_A(:,i,j);
        end
    end
    
    weights_alex = weight(weights_trans, weights_emit);
end





