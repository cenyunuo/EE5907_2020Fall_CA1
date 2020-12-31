%% Initializing 
clear;
clc;
fprintf('-->INITIALIZING...\n...\n');
load('spamData.mat');
%% Log-transformation
Xtrain = log(Xtrain+0.1);
Xtest = log(Xtest+0.1);
fprintf('-->Now all features are transformed into log(X+1).\n...\n');
%% Regularization
lambda = [1:1:9,10:5:100];
weight = zeros(length(lambda),58);
%% Obtain weight by newton method
for i = 1:length(lambda)
    weight(i,:) = newton(Xtrain,ytrain,lambda(i));
end
%% Error
error_train = zeros(1,length(lambda));
error_test = zeros(1,length(lambda));
fprintf('-->Start training.\n...\n');
for i = 1:length(lambda)
    [error_train(i), error_test(i)] = training(Xtrain, ytrain, Xtest, ytest,weight(i,:));
end
fprintf('-->Finish training.\n...\n');
fprintf('-->Training error is %f, %f, %f when lambda is 1, 10, 100.\n', [error_train(1), error_train(10), error_train(28)]);
fprintf('-->Test error is %f, %f, %f when lambda is 1, 10, 100.\n', [error_test(1), error_test(10), error_test(28)]);
%% Visualization
figure(1);
hold on;
plot(lambda,error_train,'k');
plot(lambda,error_test,'r');
title('Error funtion of \lambda');
xlabel('\lambda');
ylabel('Error');
legend('error of trainning','error of testing');
grid on;