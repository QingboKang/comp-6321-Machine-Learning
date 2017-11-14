% This is the solution for the NO.2 problem of Assignment 1
% Weighted linear regression
% Author: Qingbo Kang 
% Student ID: 40058122
% e-mail: qi_kang@encs.concordia.ca

close all;
clear;

% load the data into memory
hw1x = load('hw1x.dat');
hw1y = load('hw1y.dat');

% the number of data samples
n = size(hw1x, 1);

% weight diagonal matrix
U = diag(ones(1, n));
% find the largest and its index 
[largest, largestIndex] = max(hw1x);


% add a column vector of 1s to the inputs
hw1x = [hw1x, ones(size(hw1x, 1), 1)];

for largestWeight = 0:10
    % set the weight of the largest input value
    U(largestIndex(1), largestIndex(1)) = largestWeight;
    % obtain the weight vector w use the linear regression formula
    w = (hw1x'* U * hw1x)^-1 * hw1x' * U * hw1y;


    % plot the linear regression line with the data 
    figure
    plot(hw1x(:, end - 1), hw1y, 'r.', 'MarkerSize', 10);
    hold on
    plot(hw1x(:, end - 1), hw1x*w, 'b-' );
    legend('Training data', 'Weighted linear regression');
    title('2-c');
    hold off
end


%% d 
count = 1;
for ii = 1:n
    if( hw1x(ii, 1) >= -2 && hw1x(ii, 1) <= 2 )
        target(count, 1) = hw1x(ii, 1);
        target(count, 2) = 1;
        target(count, 3) = hw1y(ii, 1);
        count = count + 1;
        U(ii, ii) = 1;
    else 
        U(ii, ii) = 0.01;
    end
end

% obtain the weight vector w use the weighted linear regression formula
w_weighted = (hw1x'* U * hw1x)^-1 * hw1x' * U * hw1y;
% obtain the weight vector w use the linear regression formula
w = (hw1x'*hw1x)^-1 * hw1x' * hw1y;

% plot the weighted linear regression line with the data 
figure
plot(hw1x(:, end - 1), hw1y, 'r.', 'MarkerSize', 10);
hold on,
plot(target(:, 1), target(:, 3), 'g.', 'MarkerSize', 10);
plot(target(:, 1), target(:, 1:2)*w_weighted, 'k-' );
plot(hw1x(:, end - 1), hw1x*w, 'b-' );
legend('All training data', 'Local data', 'Weighted linear regression', ...
       'Linear regression');
title('2-d');
hold off