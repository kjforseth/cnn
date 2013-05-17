% used to initialize the weight matrices

function randMat = convnn_rand(X,n)

if numel(X) == 1
    randMat = (X-0.5)/(n^(1/2));
else
    randMat = (X-0.5)/(n^(1/2))/std(reshape(X-0.5,1,[]));
end