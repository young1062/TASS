classdef mini_batch
    properties 
        mb
    end
    properties (Dependent)
        n_clusters
        counts
    end
    methods
        function obj = mini_batch(mb)
            obj.mb = mb;
        end
        
        % Return how many clusters in the data:
        function n_clusters = get.n_clusters(obj)
            n_clusters = length(obj.mb);
        end
        
        % Count how many point from each cluster
        function counts = get.counts(obj)
            counts = zeros(1,obj.n_clusters);
            for i = 1:obj.n_clusters
                counts(i) = length(cell2mat(obj.mb(i)));
            end
        end
    end
end