function [error_train, error_test] = training(Xtrain, ytrain, Xtest, ytest, a)
%% Class prior
prior_0=sum(ytrain==0)/(sum(ytrain==0)+sum(ytrain~=0));
prior_1=sum(ytrain~=0)/(sum(ytrain==0)+sum(ytrain~=0));

%% Gaussian hyper parameter
[n_train, D] = size(Xtrain);
% For y = 0,
miu_0 = zeros(1,D);
miu_1 = zeros(1,D);
sigma_0 = zeros(1,D);
sigma_1 = zeros(1,D);

for i = 1:D
    miu_0(i) = sum(Xtrain(find(ytrain == 0),i))/sum(ytrain == 0);
    miu_1(i) = sum(Xtrain(find(ytrain == 1),i))/sum(ytrain == 1);
    sigma_0(i) = sum((Xtrain(find(ytrain == 0),i)-miu_0(i)).*(Xtrain(find(ytrain == 0),i)-miu_0(i)))/sum(ytrain == 0);
    sigma_1(i) = sum((Xtrain(find(ytrain == 1),i)-miu_1(i)).*(Xtrain(find(ytrain == 1),i)-miu_1(i)))/sum(ytrain == 1);
end

%% Feature likelihood

likelihood_0=zeros(n_train,1);
likelihood_1=zeros(n_train,1);
for i = 1:n_train
    likelihood_0(i) = prod(1./sqrt(2*pi*sigma_0).*exp(-(Xtrain(i,:)-miu_0).^2/2./sigma_0));
    likelihood_1(i) = prod(1./sqrt(2*pi*sigma_1).*exp(-(Xtrain(i,:)-miu_1).^2/2./sigma_1));
end

%% Bayes rule
% With bayes rule, we can deduce estimation of each class from class prior
%and feature likelihood
% Note that we do not apply log transformation here
estimation_0 = likelihood_0 * prior_0; 
estimation_1 = likelihood_1 * prior_1; 

estimation_ytrain = zeros(n_train,1); 
estimation_ytrain(find(estimation_1 > estimation_0)) = 1;


%% Testing

n_test = length(Xtest);

likelihood_0=zeros(n_test,1);
likelihood_1=zeros(n_test,1);
for i = 1:n_test
    likelihood_0(i) = prod(1./sqrt(2*pi*sigma_0).*exp(-(Xtest(i,:)-miu_0).^2/2./sigma_0));
    likelihood_1(i) = prod(1./sqrt(2*pi*sigma_1).*exp(-(Xtest(i,:)-miu_1).^2/2./sigma_1));
end

estimation_0 = likelihood_0 * prior_0;
estimation_1 = likelihood_1 * prior_1;

estimation_ytest = zeros(n_test,1);
estimation_ytest(find(estimation_1 >= estimation_0)) = 1;

%%

%calculate error
error_train = sum(abs(estimation_ytrain - ytrain)) / n_train;
error_test = sum(abs(estimation_ytest - ytest)) / n_test;


end