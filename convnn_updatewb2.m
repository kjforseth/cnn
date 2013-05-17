% weight update application

function net = convnn_updatewb2(dW,net)

uInd = 0;

for n = length(net):-1:2 % skip first dummy layer
    switch net{n}.type
        case 'conv'
            numW = prod(net{n}.kernDim);
            for m = 1:net{n}.kernNum
                convInd = find(net{n}.convMap(m,:)); % find which feature maps from previous layer to accumulate
                for o = convInd
                    net{n}.W{m,o} = net{n}.W{m,o} - reshape(dW(uInd + (1:numW)),net{n}.kernDim);
                    uInd = uInd + numW;
                end
            end
            
            numB = 1;
            for m = 1:net{n}.kernNum
                net{n}.B{m} = net{n}.B{m} - dW(uInd + (1:numB));
                uInd = uInd + numB;
            end
        case 'subsamp'
            numW = 1;
            for m = 1:net{n}.FMapsNum
                net{n}.W{m} = net{n}.W{m} - dW(uInd + (1:numW));
                uInd = uInd + numW;
            end
            
            numB = 1;
            for m = 1:net{n}.FMapsNum
                net{n}.B{m} = net{n}.B{m} - dW(uInd + (1:numB));
                uInd = uInd + numB;
            end
        case 'full'
            numW = numel(net{n}.W);
            net{n}.W = net{n}.W - reshape(dW(uInd + (1:numW)),size(net{n}.W));
            uInd = uInd + numW;
            
            numB = numel(net{n}.B);
            net{n}.B = net{n}.B - dW(uInd + (1:numB))';
            uInd  = uInd + numB;
        otherwise
            error('KJF: unknown layer type')
    end
end