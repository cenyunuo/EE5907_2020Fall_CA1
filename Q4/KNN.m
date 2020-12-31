function [error_train, error_test] = KNN(Xtrain, ytrain, Xtest, ytest, K)

    [n_test, D] = size(Xtest);
    [n_train, D] = size(Xtrain);
%% Training error
    estimation_ytrain = zeros(n_train,1); 
    for i = 1:n_train
        % Euclidean Distance
        d = euclidean(Xtrain(i,:), Xtrain);
        % Sort the point with the closest distance 
        [sort_d, index] = sort(d);
        % Capture K nearest point
        temp = ytrain(index(1:K));
        % Totally K_1 belongs to y=1 and K_0 belongs to y=0
        K_1 = sum(temp);
        K_0 = K - K_1;
        if K_1 >= K_0
            estimation_ytrain(i) = 1;
        else
            estimation_ytrain(i) = 0;
        end
    end    

    error_train = sum(abs(estimation_ytrain-ytrain)) / n_train;
%% Testting error    
    estimation_ytest = zeros(n_test,1); 

    for i = 1:n_test

        d = euclidean(Xtest(i,:), Xtrain);
        [sort_d, index] = sort(d);

        temp = ytrain(index(1:K));
        K_1 = sum(temp);
        K_0 = K - K_1;

        if K_1 >= K_0
            estimation_ytest(i) = 1;
        else
            estimation_ytest(i) = 0;
        end

    end
    
    error_test = sum(abs(estimation_ytest-ytest)) / n_test;
end