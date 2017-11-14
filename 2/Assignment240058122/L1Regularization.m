function [ w ] = L1Regularization( X, y, lambda)
%L1REGULARIZATION Using the quadprog function of Matlab to perform L1 
%                 regularization

A = [1, 1, 1, 1; 
     1, 1, 1, -1;
     1, 1, -1, 1;
     1, 1, -1, -1;
     1, -1, 1, 1;
     1, -1, 1, -1;
     1, -1, -1, 1;
     1, -1, -1, -1;
     -1, 1, 1, 1;
     -1, 1, 1, -1;
     -1, 1, -1, 1;
     -1, 1, -1, -1;
     -1, -1, 1, 1;
     -1, -1, 1, -1;
     -1, -1, -1, 1;
     -1, -1, -1, -1 ];

 H = X' * X;
 f = X' * y;
 b = ones(length(A), 1);

 w = quadprog(H, f, lambda * A, b);
 

end

