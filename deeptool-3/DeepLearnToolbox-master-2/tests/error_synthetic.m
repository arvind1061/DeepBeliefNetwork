function test_example_DBN

syndata=drawtraj([2 10],[5 15],[6 20]);
%normalizing syndata
maxval=max(syndata);
minval=min(syndata);
syn_data=(syndata-minval)/(maxval-minval);
syn_data=syn_data(:,1:700);

syn_data=double(syn_data);
err=[];
a=[];
%%  ex1 train a 100 hidden unit RBM and visualize its weights
for layers = 1:20
    er = [];   
    a = [a 1];
    for nodes = 100:100:700
        rand('state',0)
        dbn.sizes = nodes*a; 
        opts.numepochs =   1;
        opts.batchsize =   1;
        opts.momentum  =   0;
        opts.alpha     =   1;
        dbn = dbnsetup(dbn, syn_data, opts);
        [dbn, ey] = dbntrain(dbn, syn_data, opts, nodes);
        er = [er; ey];
    end
    err = [err er];
end

X = 1:20;
Y = 100:100:700;

%dlmwrite('exp_X',X,'\t');
%dlmwrite('exp_Y',Y,'\t');
dlmwrite('Output',err,'\t');
a = load('Output');
%plot(a(:))
pcolor(X,Y,a)
%figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights




%%  ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN
% rand('state',0)
% %train dbn
% dbn.sizes = [100 100];
% opts.numepochs =   1;
% opts.batchsize = 100;
% opts.momentum  =   0;
% opts.alpha     =   1;
% dbn = dbnsetup(dbn, train_x, opts);
% dbn = dbntrain(dbn, train_x, opts);
% 
% %unfold dbn to nn
% nn = dbnunfoldtonn(dbn, 10);
% nn.activation_function = 'sigm';
% 
% %train nn
% opts.numepochs =  1;
% opts.batchsize = 100;
% nn = nntrain(nn, train_x, train_y, opts);
% [er, bad] = nntest(nn, test_x, test_y);
% 
% assert(er < 0.10, 'Too big error');
