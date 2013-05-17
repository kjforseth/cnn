% define transfer function handles

function [tFcn,dtFcn] = convnn_tFcn(tFcnName)

switch tFcnName
    case 'lin'
        tFcn  = @(X) X;
        dtFcn = @(X) ones(size(X));
    case 'tanh'
        a = 1.7159;
        s = 2/3;
        tFcn  = @(X) a*tanh(s*X);
        dtFcn = @(X) a*s*(1-tanh(s*X).^2);
    case 'rbf'
        % to be added
    otherwise
        error('KJF: unrecognized transfer function')
end