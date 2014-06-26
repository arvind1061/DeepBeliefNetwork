function [dbn val] = dbntrain(dbn, x, opts, nodes)
    n = numel(dbn.rbm);
    
    [dbn.rbm{1} err] = rbmtrain(dbn.rbm{1}, x, opts, nodes);
    for i = 2 : n
        x = rbmup(dbn.rbm{i - 1}, x);
        [dbn.rbm{i} err] = rbmtrain(dbn.rbm{i}, x, opts, nodes);
    end
    val=err(end);
end
