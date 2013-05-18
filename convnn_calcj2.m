% calculate gradient with backpropagation

function [j,net] = convnn_calcj2(j,e,net,params)

j(:) = 0; % initialize row of weight/bias gradients
jInd = 0;

for n = length(net):-1:2 % skip first dummy layer
    switch net{n}.type
        case 'conv'
            net{n}.dEdY = net{n}.dEdX; % linear transfer function for conv
            
            dEdW = num2cell(zeros(net{n}.kernNum,net{n-1}.FMapsNum));
            dEdB = num2cell(zeros(net{n}.kernNum,1));
            dEdX = num2cell(zeros(1,net{n-1}.FMapsNum));
            for m = 1:net{n}.kernNum
                convInd = find(net{n}.convMap(m,:)); % find which feature maps from previous layer to accumulate
                for o = convInd
%                     dEdW{m,o} = dEdW{m,o} + conv2(net{n-1}.X{o},rot90(net{n}.dEdY{m},2),'valid');
                    dEdW{m,o} = dEdW{m,o} + rot90(conv2(net{n-1}.X{o},rot90(net{n}.dEdY{m},2),'valid'),2);
                    dEdB{m} = dEdB{m} + sum(net{n}.dEdY{m}(:));
%                     dEdX{o} = dEdX{o} + conv2(net{n}.W{m,o},net{n}.dEdY{m},'full');
                    dEdX{o} = dEdX{o} + conv2(net{n}.dEdY{m},rot90(net{n}.W{m,o},2),'full');
                end
            end
            net{n}.dEdW = dEdW;
            net{n}.dEdB = dEdB;
            net{n-1}.dEdX = dEdX';
            
            if sum(~net{n}.convMap(:)) == 0
                numW = prod(net{n}.kernDim)*net{n}.kernNum*net{n-1}.FMapsNum;
                j(jInd + (1:numW)) = reshape(cell2mat(net{n}.dEdW),[],1);
                jInd = jInd + numW;
            else
                numW = prod(net{n}.kernDim);
                for m = 1:net{n}.kernNum
                    convInd = find(net{n}.convMap(m,:)); % find which feature maps from previous layer to accumulate
                    for o = convInd
                        j(jInd + (1:numW)) = reshape(net{n}.dEdW{m,o},[],1);
                        jInd = jInd + numW;
                    end
                end
            end
            
            numB = net{n}.kernNum;
            j(jInd + (1:numB)) = cell2mat(net{n}.dEdB);
            jInd = jInd + numB;
        case 'subsamp'
            for m = 1:net{n}.FMapsNum
                net{n}.dXdY{m} = net{n}.dtFcn(net{n}.X{m});
                net{n}.dEdY{m} = net{n}.dXdY{m}.*net{n}.dEdX{m};
                net{n}.dEdW{m} = sum(net{n}.dEdY{m}(:).*net{n}.S{m}(:));
                net{n}.dEdB{m} = sum(net{n}.dEdY{m}(:));
                
%                 dEdX = conv2(upsample(upsample(net{n}.dEdY{m}.*net{n}.W{m},net{n}.sub)',net{n}.sub),ones(net{n}.sub),'full')'; % unnormalized by size of kernel?
%                 net{n-1}.dEdX{m} = dEdX(1:end-net{n}.sub+1,1:end-net{n}.sub+1);
                net{n-1}.dEdX{m} = kron(net{n}.dEdY{m}.*net{n}.W{m},ones(net{n}.sub));
            end
            
            numW = net{n}.FMapsNum;
            j(jInd + (1:numW)) = cell2mat(net{n}.dEdW');
            jInd = jInd + numW;
            
            numB = net{n}.FMapsNum;
            j(jInd + (1:numB)) = cell2mat(net{n}.dEdB');
            jInd = jInd + numB;
        case 'full'
            if n == length(net) % output layer
                net{n}.dEdX{1} = e; %params.dpFcn(e);
            else
                % already defined
            end
            net{n}.dXdY{1} = net{n}.dtFcn(net{n}.X);
            net{n}.dEdY{1} = net{n}.dEdX{1}.*net{n}.dXdY{1};
            
            if strcmp(net{n-1}.type,'conv')
                net{n}.dEdW{1} = kron(net{n}.dEdY{1},cell2mat(net{n-1}.X));
                net{n-1}.dEdX = num2cell(net{n}.dEdY{1}*net{n}.W');
                
                numW = net{n}.nodeNum*net{n-1}.FMapsNum;
                j(jInd + (1:numW)) = reshape(net{n}.dEdW{1},[],1);
                jInd = jInd + numW;
            elseif strcmp(net{n-1}.type,'subsamp')
                % undefined currently, previous layer should always be conv
            elseif strcmp(net{n-1}.type,'full')
                net{n}.dEdW{1} = kron(net{n}.dEdY{1},net{n-1}.X);
                net{n-1}.dEdX{1} = net{n}.dEdY{1}*net{n}.W';
                
                numW = net{n}.nodeNum*net{n-1}.nodeNum;
                j(jInd + (1:numW)) = reshape(net{n}.dEdW{1},[],1);
                jInd = jInd + numW;
            end
            
            net{n}.dEdB{1} = net{n}.dEdY{1};
            
            numB = net{n}.nodeNum;
            j(jInd + (1:numB)) = reshape(net{n}.dEdB{1},[],1);
            jInd = jInd + numB;
        otherwise
            error('KJF: unknown layer type')
    end
end