%% Initializing 
clear;
clc;
fprintf('-->INITIALIZING...\n...\n');
load('spamData.mat');
double(Xtrain(1,:));
%% Beta-binomial Naive Bayes 
[n_train, D] = size(Xtrain);
n_test = length(Xtest);
fprintf('-->There are totally %d training samples, %d test samples with %d features.\n...\n',[n_train, n_test, D]);
a = 1:0.5:100; %hyperparameter of prior beta(a,a)
%% Binarization
Xtrain_b = zeros(size(Xtrain));
Xtest_b = zeros(size(Xtest));
Xtrain_b(Xtrain == 0) = 0;
Xtrain_b(Xtrain ~= 0) = 1;
Xtest_b(Xtest == 0) = 0;
Xtest_b(Xtest ~= 0) = 1;
fprintf('-->Now all features in training set and test set are binarized to 0 and 1.\n...\n');

%% Training
fprintf('-->Start training.\n...\n');
for i = 1:length(a)
    [Error_train(i), Error_test(i)] = training(Xtrain_b, ytrain, Xtest_b, ytest, a(i));
end
fprintf('-->Finish training.\n...\n');
fprintf('-->Training error is %f, %f, %f when a is 1, 10, 100.\n', [Error_train(1), Error_train(19), Error_train(199)]);
fprintf('-->Test error is %f, %f, %f when a is 1, 10, 100.\n', [Error_test(1), Error_test(19), Error_test(199)]);
%% Visualization
figure(1);
hold on;
plot(a,Error_train,'k');
plot(a,Error_test,'r');
title('Error funtion of \alpha');
xlabel('\alpha');
ylabel('Error');
legend('error of trainning','error of testing');
grid on;