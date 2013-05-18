% test gradient calculation with finite differences

function debug_flag = convnn_debug(X,Y,net,gen,params)

%% set epsilon
epsi = 1e-8;
thresh = 1e-2;


%% initialize network (with large weights)
% net = cell(gen.numLayers,1);
% for n = 1:gen.numLayers
%     switch gen.layerList{n,1}
%         case 'input'
%             net{n}.type = 'subsamp';
%             net{n}.wNum = gen.numWeights(n);
%             net{n}.teta = gen.layerList{n,2}.teta;
%             
%             net{n}.X = {};
%             net{n}.Y = {};
%             net{n}.S = {};
%             
%             net{n}.sub = 1;
%             [net{n}.tFcn,net{n}.dtFcn] = convnn_tFcn('lin');
%             
%             net{n}.FMapsNum = 1; % only 1 input image allowed
%             net{n}.FMapsDim = gen.inputDim;
%             
%             net{n}.W{1} = 1; %ones(gen.inputDim);
%             net{n}.B{1} = 0; %zeros(gen.inputDim);
%         case 'conv'
%             net{n}.type = 'conv';
%             net{n}.wNum = gen.numWeights(n);
%             net{n}.teta = gen.layerList{n,2}.teta;
%             
%             net{n}.X = {};
%             
%             net{n}.kernNum = gen.layerList{n,2}.kernNum;
%             net{n}.kernDim = gen.layerList{n,2}.kernDim;
% 
%             if ischar(gen.layerList{n,2}.convMap) && ...
%                 strcmp(gen.layerList{n,2}.convMap,'all')
%                 net{n}.convMap = ones(net{n}.kernNum,net{n-1}.FMapsNum);
%             elseif ischar(gen.layerList{n,2}.convMap) && ...
%                    strcmp(gen.layerList{n,2}.convMap,'rand')
%                warning('not yet implemented')
% %                 net{n}.convMap = round(rand(net{n}.kernNum,net{n-1}.FMapsNum)-0.1); % untested
%             elseif ~ischar(gen.layerList{n,2}.convMap)
%                 net{n}.convMap = gen.layerList{n,2}.convMap;
%             end
%             
%             net{n}.FMapsNum = net{n}.kernNum;
%             net{n}.FMapsDim = net{n-1}.FMapsDim - net{n}.kernDim + 1;
%             
%             net{n}.W = cell(net{n}.kernNum,net{n-1}.FMapsNum);
%             net{n}.B = cell(net{n}.kernNum,1);
%             for m = 1:net{n}.kernNum
%                 convInd = find(net{n}.convMap(m,:)); % find which feature maps from previous layer to accumulate
%                 for o = convInd
%                     net{n}.W{m,o} = 2*rand(net{n}.kernDim)-1;
%                 end
%                 net{n}.B{m} = 2*rand()-1;
%             end
%             
%             net{n}.dEdX = cell(net{n}.FMapsNum,1);
%         case 'subsamp'
%             net{n}.type = 'subsamp';
%             net{n}.wNum = gen.numWeights(n);
%             net{n}.teta = gen.layerList{n,2}.teta;
%             
%             net{n}.X = {};
%             net{n}.Y = {};
%             net{n}.S = {};
%             
%             net{n}.sub = gen.layerList{n,2}.sub;
%             [net{n}.tFcn,net{n}.dtFcn] = convnn_tFcn(gen.layerList{n,2}.tFcn);
%             
%             net{n}.FMapsNum = net{n-1}.FMapsNum;
%             net{n}.FMapsDim = floor(net{n-1}.FMapsDim/net{n}.sub);
%             
%             net{n}.W = cell(net{n}.FMapsNum,1);
%             net{n}.B = cell(net{n}.FMapsNum,1);
%             for m = 1:net{n}.FMapsNum
%                 net{n}.W{m} = 1;
%                 net{n}.B{m} = 2*rand()-1;
%             end
%         case 'full'
%             net{n}.type = 'full';
%             net{n}.wNum = gen.numWeights(n);
%             net{n}.teta = gen.layerList{n,2}.teta;
%             
%             net{n}.X = [];
%             net{n}.Y = [];
%             
%             net{n}.nodeNum = gen.layerList{n,2}.nodeNum;
%             [net{n}.tFcn,net{n}.dtFcn] = convnn_tFcn(gen.layerList{n,2}.tFcn);
%             
%             if strcmp(net{n-1}.type,'conv')
%                 net{n}.W = 2*rand(net{n-1}.FMapsNum,net{n}.nodeNum)-1;
%                 net{n}.B = 2*rand(1,net{n}.nodeNum)-1;
%             elseif strcmp(net{n-1}.type,'subsamp')
%                 % undefined currently, previous layer should always be conv
%             elseif strcmp(net{n-1}.type,'full')
%                 net{n}.W = 2*rand(net{n-1}.nodeNum,net{n}.nodeNum)-1;
%                 net{n}.B = 2*rand(1,net{n}.nodeNum)-1;
%             end
%         case 'output'
%             net{n}.type = 'full';
%             net{n}.wNum = gen.numWeights(n);
%             net{n}.teta = gen.layerList{n,2}.teta;
%             
%             net{n}.X = [];
%             net{n}.Y = [];
%             
%             net{n}.nodeNum = gen.outputNum;
%             [net{n}.tFcn,net{n}.dtFcn] = convnn_tFcn(gen.layerList{n,2}.tFcn);
%             
%             if strcmp(net{n-1}.type,'conv')
%                 net{n}.W = 2*rand(net{n-1}.FMapsNum,net{n}.nodeNum)-1;
%                 net{n}.B = 2*rand(1,net{n}.nodeNum)-1;
%             elseif strcmp(net{n-1}.type,'subsamp')
%                 % undefined currently
%             elseif strcmp(net{n-1}.type,'full')
%                 net{n}.W = 2*rand(net{n-1}.nodeNum,net{n}.nodeNum)-1;
%                 net{n}.B = 2*rand(1,net{n}.nodeNum)-1;
%             end
%         otherwise
%             error('KJF: unknown layer type')
%     end
%     
%     % common settings
%     net{n}.dEdW{1} = 0;
%     net{n}.dEdB{1} = 0;
%     net{n}.dEdX{1} = 0;
%     net{n}.dXdY{1} = 0;
%     net{n}.dEdY{1} = 0;
%     net{n}.dYdW{1} = 0;
%     net{n}.dYdB{1} = 0;
% end


%% finite differences
dW = zeros(sum(gen.numWeights),1);
jFD = zeros(sum(gen.numWeights),1);
hW = waitbar(0,'Running finite differences');
for n = 1:100:length(dW)
    waitbar(n/length(dW))
    dWhi = dW; dWlo = dW;
    dWhi(n) = +epsi;
    dWlo(n) = -epsi;
    
    netHi = convnn_updatewb2(dWhi,net);
    netLo = convnn_updatewb2(dWlo,net);
    
    eHi = convnn_forward2(X{1},netHi) - Y(:,1)';
    eLo = convnn_forward2(X{1},netLo) - Y(:,1)';
    
    jFD(n) = (params.pFcn(eHi)-params.pFcn(eLo))/(2*epsi);
end
close(hW)


%% compare
j = zeros(sum(gen.numWeights),1);
[y,netG] = convnn_forward2(X{1},net);
e = y-Y(:,1)';
jG = convnn_calcj2(j,e,netG,params);

err = jG-jFD;

if sum(err) > thresh
    debug_flag = true;
else
    debug_flag = false;
end