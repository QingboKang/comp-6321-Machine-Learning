% This is the solution for the NO.1 problem part b and c of Assignment 2
% L1 vs. L2 Regularization
% Author: Qingbo Kang 
% Student ID: 40058122
% e-mail: qi_kang@encs.concordia.ca

close all;
clear;

%% 
% load the data files
hw2x = load('hw2x.dat');
hw2y = load('hw2y.dat');

% add a column vector of 1s to the inputs
hw2x = [hw2x, ones(size(hw2x, 1), 1)];

% split into training and test sets
data_full = [hw2x, hw2y];
[ndata, D] = size(data_full);
R = randperm(ndata);
data_train = data_full(R(1: round(0.9*ndata)), :);
R(1: round(0.9 * ndata)) = [];
data_test = data_full(R, :);

% X and y of training set
Xtrain = data_train(:, 1: end - 1);
Ytrain = data_train(:, end);

% X and y of testing set
Xtest = data_test(:, 1: end - 1);
Ytest = data_test(:, end);

% number of samples
num_samples = size(hw2x, 1);
num_features = size(hw2x, 2);

%% b 
% using the quadprog function of Matlab, write a function that performs 
% L1 regularization.
count = 1;
for lambda = 0:1:100
    % obtain weight vector perform L1 regularization
    weight(count, :) = L1Regularization(Xtrain, Ytrain, lambda);

    % RMSE on the training and test sets
    TrainRMSE(count) = RMSE( Xtrain, Ytrain, weight(count, :) );
    TestRMSE(count)= RMSE( Xtest, Ytest, weight(count, :) );
    
    count = count + 1;
end

%% c
% plot on one graph the RMSE on the training set and test set
figure;
plot([0: 1:100], TrainRMSE, 'r.', 'MarkerSize', 15);
grid on;
hold on
plot([0: 1: 100], TestRMSE, 'b.', 'MarkerSize', 15);
title('1-bc: The RMSE on the training and test set with different \lambda')
legend('RMSE on the training set','RMSE on the test set');
xlabel('\lambda');
ylabel('RMSE');
hold off;

% plot all of the weights as a function of lambda
figure;
plot( [0: 1: 100], weight(:, 1), 'LineWidth', 2);
hold on;
plot( [0: 1: 100], weight(:, 2), 'LineWidth', 2);
hold on;
plot( [0: 1: 100], weight(:, 3), 'LineWidth', 2);
hold on;
plot( [0: 1: 100], weight(:, 4), 'LineWidth', 2);
grid on;
hold off;
title('1-bc: Weights and \lambda');
legend('weight 1', 'weight 2', 'weight 3', 'weight 4');
xlabel('\lambda');