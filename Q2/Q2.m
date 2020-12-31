%% Initializing 
clear;
clc;
fprintf('-->INITIALIZING...\n...\n');
load('spamData.mat');

%% Log-transformation
[n_train, D] = size(Xtrain);
n_test = length(Xtest);
fprintf('-->There are totally %d training samples, %d test samples with %d features.\n...\n',[n_train, n_test, D]);

Xtrain = log(Xtrain+1);
Xtest = log(Xtest+1);
fprintf('-->Now all features are transformed into log(X+1).\n...\n');

%% Training
fprintf('-->Start training.\n...\n');
[Error_train, Error_test] = training(Xtrain, ytrain, Xtest, ytest);
fprintf('-->Finish training.\n...\n');
fprintf('-->Training error is %f and the test erroe is %f', [Error_train, Error_test]);