% read files from MNIST database

% specify location
cd(path.data)
fileTrainImages = 'train-images.idx3-ubyte';
fileTrainLabels = 'train-labels.idx1-ubyte';
fileTestImages = 'test-images.idx3-ubyte';
fileTestLabels = 'test-labels.idx1-ubyte';

% extract each file
fid = fopen(fileTrainLabels);
rawTrainLabels = fread(fid);
fclose(fid);
rawTrainLabels(1:8) = []; % remove info bits
Y = -ones(10,6e4);
for n = 1:6e4; Y(rawTrainLabels(n)+1,n) = 1; end % convert to PE output

fid = fopen(fileTrainImages);
rawTrainImages = fread(fid);
fclose(fid);
rawTrainImages(1:16) = []; % remove info bits
trainImages = reshape(rawTrainImages,28,28,6e4); % separate into images
trainImages = permute(trainImages,[2 1 3]); % correct orientation
trainImages = padarray(trainImages,[2 2 0],0,'both'); % pad to 32x32
X = cell(6e4,1);
for n = 1:6e4
    X{n} = trainImages(:,:,n);
    X{n} = (X{n}-mean(X{n}(:)))/std(X{n}(:)); % normalize mean and variance
end

fid = fopen(fileTestLabels);
rawTestLabels = fread(fid);
fclose(fid);
rawTestLabels(1:8) = []; % remove info bits
Yt = -ones(10,1e4);
for n = 1:1e4; Yt(rawTestLabels(n)+1,n) = 1; end % convert to PE output

fid = fopen(fileTestImages);
rawTestImages = fread(fid);
fclose(fid);
rawTestImages(1:16) = []; % remove info bits
testImages = reshape(rawTestImages,28,28,1e4); % separate into images
testImages = permute(testImages,[2 1 3]); % correct orientation
testImages = padarray(testImages,[2 2 0],0,'both'); % pad to 32x32
Xt = cell(1e4,1);
for n = 1:1e4
    Xt{n} = testImages(:,:,n);
    Xt{n} = (Xt{n}-mean(Xt{n}(:)))/std(Xt{n}(:)); % normalize mean and variance
end

% save in .mat format
save('MNIST_trnI','X')
save('MNIST_trnL','Y')
save('MNIST_tstI','Xt')
save('MNIST_tstL','Yt')