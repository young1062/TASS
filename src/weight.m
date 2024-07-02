classdef weight
    properties 
        trans
        emit
    end
    methods 
        function obj = weight(weight_trans, weight_emit)
            obj.trans = weight_trans;
            obj.emit = weight_emit;
        end
    end
end