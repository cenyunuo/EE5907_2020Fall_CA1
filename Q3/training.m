function [error_train, error_test] = training(Xtrain, ytrain, Xtest, ytest, weight)
%% Adding dimension for regularization term
[n_train, D] = size(Xtrain);
n_test = length(Xtest);
Xtrain = [ones(n_train,1),Xtrain];
Xtest = [ones(n_test,1),Xtest];

%% miu
miu_train = 1./(1+exp(-Xtrain*weight'));
miu_test = 1./(1+exp(-Xtest*weight'));

%% Estimation
estimation_train = miu_train>=0.5;
estimation_test = miu_test>=0.5;

%% error
error_train = sum(abs(estimation_train - ytrain)) / n_train;
error_test = sum(abs(estimation_test - ytest)) / n_test;
end