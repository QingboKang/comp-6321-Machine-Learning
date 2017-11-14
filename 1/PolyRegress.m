function [ weight, x_augment ] = PolyRegress( x, y, d )
%POLYREGRESS adds the features x^2, x^3, ..., x^d to the inputs 
%      and performs polynomial regression

% add a column vector of 1s to the x
if(size(x, 2) == 1)
    x = [x, ones(size(x, 1), 1)];
end

% adds features to the x
for ii = 2:d
    x = [ x(:,end-1).^ii, x];
end

x_augment = x;

% performs polynomial regression, get the weight vector
weight = (x'*x) \ (x' * y);

end

