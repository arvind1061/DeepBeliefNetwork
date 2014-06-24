function test_example_DBN
%load mnist_uint8;
syndata=drawtraj([2 10],[5 15],[6 20]);
%normalizing syndata
maxval=max(syndata);
minval=min(syndata);
syn_data=(syndata-minval)/(maxval-minval);
syn_data=syn_data(:,1:700);
%train_x = double(train_x) / 255;
%test_x  = double(test_x)  / 255;
%train_y = double(train_y);
%test_y  = double(test_y);
syn_data=double(syn_data);
%hist(syn_data)
range=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1];
bincount=histc(syn_data,range);
prob=[];
entrinput=0.0;
% calcualting entropy of input data set
for i = 1:700;
    if syn_data(i)>=0 && syn_data(i)<0.1;
       prob(i) = bincount(1)/700;
    elseif syn_data(i)>=0.1 && syn_data(i)<0.2;
        prob(i)=bincount(2)/700;
    elseif syn_data(i)>=0.2 && syn_data(i)<0.3;
       prob(i)=bincount(3)/700;
    elseif syn_data(i)>=0.3 && syn_data(i)<0.4;
       prob(i)=bincount(4)/700;
    elseif syn_data(i)>=0.5 && syn_data(i)<0.6;
       prob(i)=bincount(5)/700;
    elseif syn_data(i)>=0.6 && syn_data(i)<0.7;
       prob(i)=bincount(6)/700;
    elseif syn_data(i)>=0.7 && syn_data(i)<0.8;
       prob(i)=bincount(7)/700;
       
    elseif syn_data(i)>=0.8 && syn_data(i)<0.9;
       prob(i)=bincount(8)/700;
    elseif syn_data(i)>=0.9 && syn_data(i)<=1.0;
       prob(i)=bincount(9)/700;
    end
    if prob(i)==0;
        continue;
    else
        x =  prob(i)*log2(prob(i));
        x = -x;
        entrinput=entrinput+x;
    end
    
end

%%  ex1 train a 100 hidden unit RBM and visualize its weights
syn_data=syn_data';
rand('state',0)
dbn.sizes = [100 100 100 100 100 100 100 100 100 100];
opts.numepochs =   1;
opts.batchsize = 100;
opts.momentum  =   0;
opts.alpha     =   1;
dbn = dbnsetup(dbn, syn_data, opts);
dbn = dbntrain(dbn, syn_data, opts);

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
