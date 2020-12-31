%% Initializing 
clear;
clc;
fprintf('-->INITIALIZING...\n...\n');
load('spamData.mat');
%% Log-transformation
Xtrain = log(Xtrain+0.1);
Xtest = log(Xtest+0.1);
fprintf('-->Now all features are transformed into log(X+0.1).\n...\n');
%% Initialize KNN parameter
K = [1:9,10:5:100];
fprintf('-->Start training.\n...\n');
error_train = zeros(1,length(K));
error_test = zeros(1,length(K));
for i = 1: length(K)
    [error_train(i), error_test(i)] = KNN(Xtrain, ytrain, Xtest, ytest, K(i));
end
fprintf('-->Finish training.\n...\n');
fprintf('-->Training error is %f, %f, %f when K is 1, 10, 100.\n', [error_train(1), error_train(10), error_train(28)]);
fprintf('-->Test error is %f, %f, %f when K is 1, 10, 100.\n', [error_test(1), error_test(10), error_test(28)]);
%% Visualization
figure(1);
hold on;
plot(K,error_train,'k');
plot(K,error_test,'r');
title('Error funtion of K');
xlabel('K');
ylabel('Error');
legend('error of trainning','error of testing');
grid on;