% convnn forward propagation

function [out,net] = convnn_forward2(in,net)

for n = 1:length(net)
    switch net{n}.type
        case 'conv'
            Y = num2cell(zeros(net{n}.kernNum,1));
            for m = 1:net{n}.kernNum
                convInd = find(net{n}.convMap(m,:)); % find which feature maps from previous layer to accumulate
                for o = convInd
%                     Y{m} = Y{m} + conv2(net{n-1}.X{o},rot90(net{n}.W{m,o},2),'valid') + net{n}.B{m};
                    Y{m} = Y{m} + conv2(net{n-1}.X{o},net{n}.W{m,o},'valid') + net{n}.B{m};
                end
            end
            net{n}.X = Y; % linear tFcn for all conv layers
        case 'subsamp'
            for m = 1:net{n}.FMapsNum
                if n == 1 % first layer
                    X = in;
                else
                    X = net{n-1}.X{m};
                end
%                 S = conv2(X,ones(net{n}.sub)/(net{n}.sub^2),'valid');
                S = conv2(X,ones(net{n}.sub),'valid'); % unnormalized by size of subsampling kernel
                net{n}.S{m} = S(1:net{n}.sub:end,1:net{n}.sub:end);
                net{n}.Y{m} = net{n}.S{m}*net{n}.W{m}+net{n}.B{m};
                net{n}.X{m} = net{n}.tFcn(net{n}.Y{m});
            end
        case 'full'
            if strcmp(net{n-1}.type,'conv')
                net{n}.Y = cell2mat(net{n-1}.X)'*net{n}.W + net{n}.B;
            elseif strcmp(net{n-1}.type,'subsamp')
                % undefined currently, previous layer should always be conv
            elseif strcmp(net{n-1}.type,'full')
                net{n}.Y = net{n-1}.X*net{n}.W + net{n}.B;
            end
            net{n}.X = net{n}.tFcn(net{n}.Y);
        otherwise
            error('KJF: unknown layer type')
    end
end

out = net{end}.X;