clear; clc;

%% Pre-processing
trainingSet = readmatrix("training_set.csv");
validationSet = readmatrix("validation_set.csv");

[trainingData,targetTrainingData] = DataPrep(trainingSet);
[validationData,targetValidationData] = DataPrep(validationSet);

M = 10;
learningRate = 0.006;
nEpoch = 10.^5;

%% Initialization
w = 2.*randn(M,2); % std = 2
W = M.*randn(1,M); % std = m
theta1 = zeros(1,M);
theta2 = zeros(1,1);

V = zeros(1,M);
O = zeros(1,1);

batchesSize = 20;
minC = 1;

%% Training
for epoch = 1:nEpoch
    for iteration = 1:size(trainingData,1)

        selectedData = fix(rand()*size(trainingData,1))+1; 

        % Vi
        for i = 1:size(V,2)
            y= w(i,:)*trainingData(selectedData,:)';
            V(i) = tanh(y-theta1(i));
        end

        % Oi
        O = tanh((W*V')-theta2);

        %Error
        deltaErrorOut = (targetTrainingData(selectedData) - O )*(1-O.^2);

        deltaErrorIn = zeros(1,M);
        for i = 1:size(V,2)
            deltaErrorIn(i) = deltaErrorOut*W(i)*(1-V(1,i).^2);
        end

        %Update weight
        W = W + learningRate*(deltaErrorOut*V);
        w = w + learningRate*(deltaErrorIn'*trainingData(selectedData,:));

        %Update threshold
        theta1 = theta1 - learningRate*deltaErrorIn;
        theta2 = theta2 - learningRate*deltaErrorOut;

        H = 0.5*(targetTrainingData(selectedData)-O).^2;
    end

    %calculate error
    valSize = size(validationData,1);
    out = zeros(1,valSize);
    Vz = zeros(2,M);
    Oz = zeros(1,1);
    C = 0;
    for data = 1:valSize
    
        % Vi
        for i = 1:size(V,2)
            y= w(i,:)*validationData(data,:)';
            V(i) = tanh(y-theta1(i));
        end

        % Oi
        Oz = tanh((W*V')-theta2);

        out(data) = sign(Oz);
%     out(data) = PropagateNetwork(validationData,data,M,w,W,theta1,theta2);
        C = C + abs(out(data)-targetValidationData(data));
    end
    C = C/(2*valSize);
    if(minC > C)
        minC = C;
    end
    sprintf('Epoch : %i, C: %0.4f min C :%0.4f',epoch,C,minC)
    if(C<0.12)
        break
    end

end
%% Validation

%calculate error
valSize = size(validationData,1);
out = zeros(1,valSize);
Vz = zeros(2,M);
Oz = zeros(1,1);
C = 0;
for data = 1:valSize

    % Vi
    for i = 1:size(V,2)
        y= w(i,:)*validationData(data,:)';
        V(i) = tanh(y-theta1(i));
    end

    % Oi
    Oz = tanh((W*V')-theta2);

    out(data) = sign(Oz);
    C = C + abs(out(data)-targetValidationData(data));
end
C = C/(2*valSize)


%% Function


function [normalizedData,target] = DataPrep(data)
    meanData = mean(data);
    stdData = std(data);
    normalizedData = (data(:,[1 2]) - meanData(:,[1 2]))./ stdData(:,[1 2]);
    target = data(:,3);
end

