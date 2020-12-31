function w =  newton(Xtrain,ytrain,lambda)
%% Adding dimension for regularization term
[n_train, D] = size(Xtrain);
w = zeros(1,D+1);
Xtrain = [ones(n_train,1),Xtrain];
temp = 2; % giving any number larger than 0(for the convience of while loop)
%% Exclude bias from l2 regularization
reg_g = ones(D+1,1);
reg_g(1,1)=0;
reg_H = diag(ones(1,D+1));
reg_H(1,1)=0;
%% while loop
i = 0;
while temp > 1e-4
    miu = 1./(1+exp(-Xtrain*w'));
    S = diag(miu.*(1-miu));
    hessian = Xtrain'*S*Xtrain + lambda*reg_H;
    gradient = Xtrain'*(miu-ytrain) +lambda*reg_g.*w';
    % Update the weight
    w = w - (inv(hessian)*gradient)';
    % Compare two weight and break the loop
    temp = norm(inv(hessian)*gradient);
    i=i+1;
end