classdef cluster
    properties 
        trans
        emit
    end
    methods 
        function obj = cluster(cluster_trans, cluster_emit)
            obj.trans = cluster_trans;
            obj.emit = cluster_emit;
        end
    end
end