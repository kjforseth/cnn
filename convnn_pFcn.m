% calculate MSE

function [mse,dmse] = convnn_pFcn(pFcn)

switch pFcn
    case 'mse'
        mse  = @(e) sum(e.^2)/length(e);
        dmse = @(e) 2*e/length(e); % maybe should be 2*sum(e)/length(e)?
    otherwise
        error('KJF: unrecognized performance function')
end