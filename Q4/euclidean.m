function [d] = euclidean(A, B) 
    [N, D] = size(B);
    d = zeros(N,1);
    d = sqrt( sum((A - B).^2 , 2));
end