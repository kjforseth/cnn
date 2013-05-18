% experimental driver for convolution neural network

%% setup
clc, clear, close all
path.root       = 'D:\matlab\cyberonics\convnn\';
path.scripts    = [path.root 'scripts\'];
path.data       = [path.root 'data\'];
path.figs       = [path.root 'figures\'];
% addpath(genpath(path.root))

% set rand seed
rng(0);


%% load data
% convnn_MNIST_reader

cd(path.data)
load('MNIST_trnI') % X: train images
load('MNIST_trnL') % Y: train labels
load('MNIST_tstI') % Xt: test images
load('MNIST_tstL') % Yt: test labels

%% training parameters
% hessian params
hes.calcMode = true;
hes.calcNum = 500;

% learning params
params.mu = 0.02;
params.epochNum = 5;
params.epochDim = 1000;
params.teta = [50 50 20 20 20 10 10 10 5 5 5 5 1 1 1 1 1 1 1 1]/1e5;

% error evaluation params
[params.pFcn,params.dpFcn] = convnn_pFcn('mse');
params.MCRcalc = 1000;
params.MCRsamps = 100;


%% characterize network
% format for input/output
gen.inputNum = 1;
gen.inputDim = [32 32];
gen.outputNum = 10;

% layer features
convMap2 = [1 0 0 0 1 1 1 0 0 1 1 1 1 0 1 1;
            1 1 0 0 0 1 1 1 0 0 1 1 1 1 0 1;
            1 1 1 0 0 0 1 1 1 0 0 1 0 1 1 1;
            0 1 1 1 0 0 1 1 1 1 0 0 1 0 1 1;
            0 0 1 1 1 0 0 1 1 1 1 0 1 1 0 1; 
            0 0 0 1 1 1 0 0 1 1 1 1 0 1 1 1; 
           ]'; % kernNum in current conv layer by FMapsNum in previous subsamp layer

gen.layerList = {'input'    ,struct('teta',0.2)
                 'conv'     ,struct('teta',0.2,'kernNum',6,'kernDim',[5 5],'convMap','all') % convmap: all,rand,var
                 'subsamp'  ,struct('teta',0.2,'tFcn','tanh','sub',2)
                 'conv'     ,struct('teta',0.2,'kernNum',16,'kernDim',[5 5],'convMap',convMap2)
                 'subsamp'  ,struct('teta',0.2,'tFcn','tanh','sub',2)
                 'conv'     ,struct('teta',0.2,'kernNum',120,'kernDim',[5 5],'convMap','all')
                 'full'     ,struct('teta',0.2,'tFcn','tanh','nodeNum',84)
                 'output'   ,struct('teta',0.2,'tFcn','tanh')
                 };
gen.numLayers = size(gen.layerList,1);
gen.numCLayers = 0; gen.numSLayers = 0; gen.numFLayers = 0;
gen.numWeights = zeros(length(gen.layerList),1);
for n = 1:gen.numLayers
    switch gen.layerList{n,1}
        case 'input'
            gen.numWeights(n) = 0; %gen.inputNum*(prod(gen.inputDim) + 1);
        case 'conv'
            gen.numCLayers = gen.numCLayers + 1;
            if ischar(gen.layerList{n,2}.convMap) && ...
               strcmp(gen.layerList{n,2}.convMap,'all')
                if n <= 2 %
                    gen.numWeights(n) = gen.inputNum*gen.layerList{n,2}.kernNum*prod(gen.layerList{n,2}.kernDim) + gen.layerList{n,2}.kernNum;
                else
                    gen.numWeights(n) = gen.layerList{n-2,2}.kernNum*gen.layerList{n,2}.kernNum*prod(gen.layerList{n,2}.kernDim) + gen.layerList{n,2}.kernNum;
                end
            elseif ischar(gen.layerList{n,2}.convMap) && ...
                   strcmp(gen.layerList{n,2}.convMap,'rand')
               warning('not yet implemented')
            elseif ~ischar(gen.layerList{n,2}.convMap)
                gen.numWeights(n) = sum(gen.layerList{n,2}.convMap(:))*prod(gen.layerList{n,2}.kernDim) + gen.layerList{n,2}.kernNum;
            end
        case 'subsamp'
            gen.numSLayers = gen.numSLayers + 1;
            gen.numWeights(n) = gen.layerList{n-1,2}.kernNum*2;
        case 'full'
            gen.numFLayers = gen.numFLayers + 1;
            if strcmp('conv',gen.layerList{n-1,1})
                gen.numWeights(n) = (gen.layerList{n-1,2}.kernNum + 1)*gen.layerList{n,2}.nodeNum;
            elseif strcmp('subsamp',gen.layerList{n-1,1})
                % conv layer should always precede full layer
            elseif strcmp('full',gen.layerList{n-1,1})
                gen.numWeights(n) = (gen.layerList{n-1,2}.nodeNum + 1)*gen.layerlist{n,2}.nodeNum;
            end
        case 'output'
            gen.numWeights(n) = (gen.layerList{n-1,2}.nodeNum + 1)*gen.outputNum;
        otherwise
            error('KJF: unknown layer type')
    end
end


%% initialize network
net = cell(gen.numLayers,1);
for n = 1:gen.numLayers
    switch gen.layerList{n,1}
        case 'input'
            net{n}.type = 'subsamp';
            net{n}.wNum = gen.numWeights(n);
            net{n}.teta = gen.layerList{n,2}.teta;
            
            net{n}.X = {};
            net{n}.Y = {};
            net{n}.S = {};
            
            net{n}.sub = 1;
            [net{n}.tFcn,net{n}.dtFcn] = convnn_tFcn('lin');
            
            net{n}.FMapsNum = 1; % only 1 input image allowed
            net{n}.FMapsDim = gen.inputDim;
            
            net{n}.W{1} = 1; %ones(gen.inputDim);
            net{n}.B{1} = 0; %zeros(gen.inputDim);
        case 'conv'
            net{n}.type = 'conv';
            net{n}.wNum = gen.numWeights(n);
            net{n}.teta = gen.layerList{n,2}.teta;
            
            net{n}.X = {};
            
            net{n}.kernNum = gen.layerList{n,2}.kernNum;
            net{n}.kernDim = gen.layerList{n,2}.kernDim;

            if ischar(gen.layerList{n,2}.convMap) && ...
                strcmp(gen.layerList{n,2}.convMap,'all')
                net{n}.convMap = ones(net{n}.kernNum,net{n-1}.FMapsNum);
            elseif ischar(gen.layerList{n,2}.convMap) && ...
                   strcmp(gen.layerList{n,2}.convMap,'rand')
               warning('not yet implemented')
%                 net{n}.convMap = round(rand(net{n}.kernNum,net{n-1}.FMapsNum)-0.1); % untested
            elseif ~ischar(gen.layerList{n,2}.convMap)
                net{n}.convMap = gen.layerList{n,2}.convMap;
            end
            
            net{n}.FMapsNum = net{n}.kernNum;
            net{n}.FMapsDim = net{n-1}.FMapsDim - net{n}.kernDim + 1;
            
            net{n}.W = cell(net{n}.kernNum,net{n-1}.FMapsNum);
            net{n}.B = cell(net{n}.kernNum,1);
            for m = 1:net{n}.kernNum
                convInd = find(net{n}.convMap(m,:)); % find which feature maps from previous layer to accumulate
                for o = convInd
                    net{n}.W{m,o} = convnn_rand(rand(net{n}.kernDim),prod(net{n}.kernDim)*sum(net{n}.convMap(:)));
                end
                net{n}.B{m} = convnn_rand(rand(),prod(net{n}.kernDim)*sum(net{n}.convMap(:)));
            end
            
            net{n}.dEdX = cell(net{n}.FMapsNum,1);
        case 'subsamp'
            net{n}.type = 'subsamp';
            net{n}.wNum = gen.numWeights(n);
            net{n}.teta = gen.layerList{n,2}.teta;
            
            net{n}.X = {};
            net{n}.Y = {};
            net{n}.S = {};
            
            net{n}.sub = gen.layerList{n,2}.sub;
            [net{n}.tFcn,net{n}.dtFcn] = convnn_tFcn(gen.layerList{n,2}.tFcn);
            
            net{n}.FMapsNum = net{n-1}.FMapsNum;
            net{n}.FMapsDim = floor(net{n-1}.FMapsDim/net{n}.sub);
            
            net{n}.W = cell(net{n}.FMapsNum,1);
            net{n}.B = cell(net{n}.FMapsNum,1);
            for m = 1:net{n}.FMapsNum
                net{n}.W{m} = 1;
                net{n}.B{m} = convnn_rand(rand(),1);
            end
        case 'full'
            net{n}.type = 'full';
            net{n}.wNum = gen.numWeights(n);
            net{n}.teta = gen.layerList{n,2}.teta;
            
            net{n}.X = [];
            net{n}.Y = [];
            
            net{n}.nodeNum = gen.layerList{n,2}.nodeNum;
            [net{n}.tFcn,net{n}.dtFcn] = convnn_tFcn(gen.layerList{n,2}.tFcn);
            
            if strcmp(net{n-1}.type,'conv')
                net{n}.W = convnn_rand(rand(net{n-1}.FMapsNum,net{n}.nodeNum),net{n-1}.FMapsNum+1);
                net{n}.B = convnn_rand(rand(1,net{n}.nodeNum),net{n-1}.FMapsNum+1);
            elseif strcmp(net{n-1}.type,'subsamp')
                % undefined currently, previous layer should always be conv
            elseif strcmp(net{n-1}.type,'full')
                net{n}.W = convnn_rand(rand(net{n-1}.nodeNum,net{n}.nodeNum),net{n-1}.nodeNum+1);
                net{n}.B = convnn_rand(rand(1,net{n}.nodeNum),net{n-1}.nodeNum+1);
            end
        case 'output'
            net{n}.type = 'full';
            net{n}.wNum = gen.numWeights(n);
            net{n}.teta = gen.layerList{n,2}.teta;
            
            net{n}.X = [];
            net{n}.Y = [];
            
            net{n}.nodeNum = gen.outputNum;
            [net{n}.tFcn,net{n}.dtFcn] = convnn_tFcn(gen.layerList{n,2}.tFcn);
            
            if strcmp(net{n-1}.type,'conv')
                net{n}.W = convnn_rand(rand(net{n-1}.FMapsNum,net{n}.nodeNum),net{n-1}.FMapsNum+1);
                net{n}.B = convnn_rand(rand(1,net{n}.nodeNum),net{n-1}.FMapsNum+1);
            elseif strcmp(net{n-1}.type,'subsamp')
                % undefined currently
            elseif strcmp(net{n-1}.type,'full')
                net{n}.W = convnn_rand(rand(net{n-1}.nodeNum,net{n}.nodeNum),net{n-1}.nodeNum+1);
                net{n}.B = convnn_rand(rand(1,net{n}.nodeNum),net{n-1}.nodeNum+1);
            end
        otherwise
            error('KJF: unknown layer type')
    end
    
    % common settings
    net{n}.dEdW{1} = 0;
    net{n}.dEdB{1} = 0;
    net{n}.dEdX{1} = 0;
    net{n}.dXdY{1} = 0;
    net{n}.dEdY{1} = 0;
    net{n}.dYdW{1} = 0;
    net{n}.dYdB{1} = 0;
end


%% validate network
% debug_flag = convnn_debug(X,Y,net,gen,params);

%% train network
% initialize
I1 = sparse(1:sum(gen.numWeights),1:sum(gen.numWeights),ones(1,sum(gen.numWeights)));
I2 = sparse(0);

j = zeros(sum(gen.numWeights),1);
h = zeros(sum(gen.numWeights),1);

% march
mcrTrn = zeros(floor(params.epochNum*params.epochDim/params.MCRcalc),1);
mcrTst = zeros(floor(params.epochNum*params.epochDim/params.MCRcalc),1);
rmse = zeros(params.epochNum*params.epochDim,1);
for n = 1:params.epochNum
    if hes.calcMode
        hW = waitbar(0,sprintf('Evaluating Hessian'));
        sInd = randperm(length(X));
        for m = 1:hes.calcNum
            waitbar(m/hes.calcNum)

            [y,net] = convnn_forward2(X{sInd(m)},net);
            e = y-Y(:,sInd(m))';

            [j,net] = convnn_calcj2(j,e,net,params);
            [h,net] = convnn_calch2(h,net);

            I2 = I2 + diag(sparse(h));
        end
        I2 = I2/hes.calcNum;
        close(hW)
    end
    
    hW = waitbar(0,sprintf('Running Epoch %d/%d',n,params.epochNum));
    sInd = randperm(length(X));
    for m = 1:params.epochDim
        waitbar(m/params.epochDim)
        
        [y,net] = convnn_forward2(X{sInd(m)},net);
        e = y-Y(:,sInd(m))';
        
        [j,net] = convnn_calcj2(j,e,net,params);
        if hes.calcMode
            dW = (I2 + params.mu*I1)\(params.teta(n)*j);
        else
            dW = params.teta(n)*j;
        end

        net = convnn_updatewb2(dW,net);
        
        pInd = (n-1)*params.epochDim+m;
        rmse(pInd) = sqrt(mean(params.pFcn(e).^2));
        if mod(m,params.MCRcalc) == 0
            mcrInd = randperm(length(X));
            c = 0;
            for o = 1:params.MCRsamps
                y = convnn_forward2(X{mcrInd(o)},net);
                [~,yi] = max(y);
                [~,Yi] = max(Y(:,mcrInd(o)));
                c = c + (yi == Yi);
            end
            mcrTrn(pInd/params.MCRcalc) = 1 - c/params.MCRsamps;
            
            mcrInd = randperm(length(Xt));
            c = 0;
            for o = 1:params.MCRsamps
                y = convnn_forward2(Xt{mcrInd(o)},net);
                [~,yi] = max(y);
                [~,Yi] = max(Yt(:,mcrInd(o)));
                c = c + (yi == Yi);
            end
            mcrTst(pInd/params.MCRcalc) = 1 - c/params.MCRsamps;
            
            fprintf('Train MCR: %0.2f\t Test MCR: %0.2f\t\t Iterations %d\n',...
                mcrTrn(pInd/params.MCRcalc),mcrTst(pInd/params.MCRcalc),pInd)
            
            figure(1)
            plot(1:pInd,rmse(1:pInd),'-b')
        end
    end
    close(hW)
end

plot(1:params.epochNum*params.epochDim,rmse)