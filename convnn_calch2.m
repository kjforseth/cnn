% diagonal approximation of hessian

function [h,net] = convnn_calch2(h,net)

h(:) = 0; % initialize hessian
hInd = 0;

for n = length(net):-1:2 % skip first dummy layer
    switch net{n}.type
        case 'conv'
            net{n}.d2Ed2Y = net{n}.d2Ed2X; % linear conv operator
            
            d2Ed2W = num2cell(zeros(net{n}.kernNum,net{n-1}.FMapsNum));
            d2Ed2B = num2cell(zeros(net{n}.kernNum,1));
            d2Ed2X = num2cell(zeros(1,net{n-1}.FMapsNum));
            for m = 1:net{n}.kernNum
                convInd = find(net{n}.convMap(m,:)); % find which feature maps from previous layer to accumulate
                for o = convInd
%                     d2Ed2W{m,o} = d2Ed2W{m,o} + conv2(net{n-1}.X{o}.^2,rot90(net{n}.d2Ed2Y{m},2),'valid');
                    d2Ed2W{m,o} = d2Ed2W{m,o} + rot90(conv2(net{n-1}.X{o}.^2,rot90(net{n}.d2Ed2Y{m},2),'valid'),2);
                    d2Ed2B{m} = d2Ed2B{m} + sum(net{n}.d2Ed2Y{m}(:));
%                     d2Ed2X{o} = d2Ed2X{o} + conv2(net{n}.W{m,o}.^2,net{n}.d2Ed2Y{m},'full');
                    d2Ed2X{o} = d2Ed2X{o} + conv2(net{n}.d2Ed2Y{m},rot90(net{n}.W{m,o}.^2,2),'full');
                end
            end
            net{n}.d2Ed2W = d2Ed2W;
            net{n}.d2Ed2B = d2Ed2B;
            net{n-1}.d2Ed2X = d2Ed2X;
            
            if sum(~net{n}.convMap(:)) == 0
                numW = prod(net{n}.kernDim)*net{n}.kernNum*net{n-1}.FMapsNum;
                h(hInd + (1:numW)) = reshape(cell2mat(net{n}.d2Ed2W),[],1);
                hInd = hInd + numW;
            else
                numW = prod(net{n}.kernDim);
                for m = 1:net{n}.kernNum
                    convInd = find(net{n}.convMap(m,:)); % find which feature maps from previous layer to accumulate
                    for o = convInd
                        h(hInd + (1:numW)) = reshape(net{n}.d2Ed2W{m,o},[],1);
                        hInd = hInd + numW;
                    end
                end
            end
            
            numB = net{n}.kernNum;
            h(hInd + (1:numB)) = cell2mat(net{n}.d2Ed2B);
            hInd = hInd + numB;
        case 'subsamp'
            for m = 1:net{n}.FMapsNum
                net{n}.d2Ed2Y{m} = net{n}.d2Ed2X{m}.*(net{n}.dXdY{m}.^2);
                d2Ed2W = net{n}.d2Ed2Y{m}*(net{n}.S{m}.^2);
                net{n}.d2Ed2W{m} = sum(d2Ed2W(:));
                net{n}.d2Ed2B{m} = sum(net{n}.d2Ed2Y{m}(:));
                
%                 d2Ed2X = conv2(upsample(upsample(net{n}.d2Ed2Y{m}.*(net{n}.W{m}.^2),net{n}.sub)',net{n}.sub),ones(net{n}.sub),'full')'; % unnormalized by size of kernel?
%                 net{n-1}.d2Ed2X{m} = d2Ed2X(1:end-net{n}.sub+1,1:end-net{n}.sub+1);
                net{n-1}.d2Ed2X{m} = kron(net{n}.d2Ed2Y{m}.*(net{n}.W{m}.^2),ones(net{n}.sub));
            end
            
            numW = net{n}.FMapsNum;
            h(hInd + (1:numW)) = cell2mat(net{n}.d2Ed2W');
            hInd = hInd + numW;
            
            numB = net{n}.FMapsNum;
            h(hInd + (1:numB)) = cell2mat(net{n}.d2Ed2B');
            hInd = hInd + numB;
        case 'full'
            if n == length(net) % output layer
                net{n}.d2Ed2X{1} = 1; % 2nd deriv of MSE is 1
            else
                % already defined
            end
            
            net{n}.d2Ed2Y{1} = net{n}.d2Ed2X{1}.*(net{n}.dXdY{1}.^2);
            
            if strcmp(net{n-1}.type,'conv')
                net{n}.d2Ed2W{1} = kron(net{n}.d2Ed2Y{1},cell2mat(net{n-1}.X).^2);
                net{n-1}.d2Ed2X = num2cell(net{n}.d2Ed2Y{1}*(net{n}.W.^2)'); % num2cell()?, transpose?
                
                numW = net{n}.nodeNum*net{n-1}.FMapsNum;
                h(hInd + (1:numW)) = reshape(net{n}.d2Ed2W{1},[],1);
                hInd = hInd + numW;
            elseif strcmp(net{n-1}.type,'subsamp')
                % undefined currently, previous layer should always be conv
            elseif strcmp(net{n-1}.type,'full')
                net{n}.d2Ed2W{1} = kron(net{n}.d2Ed2Y{1},net{n-1}.X.^2);
                net{n-1}.d2Ed2X{1} = net{n}.d2Ed2Y{1}*(net{n}.W.^2)';
                
                numW = net{n}.nodeNum*net{n-1}.nodeNum;
                h(hInd + (1:numW)) = reshape(net{n}.d2Ed2W{1},[],1); 
                hInd = hInd + numW;
            end
            
            net{n}.d2Ed2B{1} = net{n}.d2Ed2Y{1};
            
            numB = net{n}.nodeNum;
            h(hInd + (1:numB)) = reshape(net{n}.d2Ed2B{1},[],1);
            hInd = hInd + numB;
        otherwise
            error('KJF: unknown layer type')
    end
end