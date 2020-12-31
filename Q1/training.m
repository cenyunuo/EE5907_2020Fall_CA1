function [error_train, error_test] = training(Xtrain_b, ytrain, Xtest_b, ytest, a)
%% Class prior
prior_0=sum(ytrain==0)/(sum(ytrain==0)+sum(ytrain~=0));
prior_1=sum(ytrain~=0)/(sum(ytrain==0)+sum(ytrain~=0));

%% Feature likelihood
[n_train, D] = size(Xtrain_b);
theta_0 = zeros(1,D);
theta_1 = zeros(1,D);

for i = 1:D
    %y=0
    N_0 = sum(ytrain == 0);
    N_0_1 = sum(Xtrain_b(find(ytrain == 0),i));
    theta_0(i) = (N_0_1 + a - 1) / (N_0 + a + a - 2);
    %y=1
    N_1 = sum(ytrain == 1);
    N_1_1 = sum(Xtrain_b(find(ytrain == 1),i));
    theta_1(i) = (N_1_1 + a - 1) / (N_1 + a + a - 2);
end

%When y = 0, the feature likelihood is:
likelihood_0 = Xtrain_b .* theta_0 + (1-Xtrain_b) .* (1-theta_0);
%Likewise, when y = 1:
likelihood_1 = Xtrain_b .* theta_1 + (1-Xtrain_b) .* (1-theta_1);

%% Bayes rule
% With bayes rule, we can deduce estimation of each class from class prior
%and feature likelihood
% Note that we do not apply log transformation here
estimation_0 = prod(likelihood_0,2) * prior_0; 
estimation_1 = prod(likelihood_1,2) * prior_1; 

estimation_ytrain = zeros(n_train,1); 
estimation_ytrain(find(estimation_1 > estimation_0)) = 1;


%% Testing

% Now applying the prior and MAP

[n_test, D] = size(Xtest_b);


likelihood_0_ = Xtest_b .* theta_0 + (1-Xtest_b) .* (1-theta_0);
likelihood_1_ = Xtest_b .* theta_1 + (1-Xtest_b) .* (1-theta_1);

estimation_0_ = prod(likelihood_0_,2) * prior_0;
estimation_1_ = prod(likelihood_1_,2) * prior_1;


estimation_ytest = zeros(n_test,1);
estimation_ytest(find(estimation_1_ >= estimation_0_)) = 1;

%%

%calculate error
error_train = sum(abs(estimation_ytrain - ytrain)) / n_train;
error_test = sum(abs(estimation_ytest - ytest)) / n_test;


end