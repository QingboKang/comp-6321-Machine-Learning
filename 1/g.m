close all;

% load the data into memory and plot it
hw1x = load('hw1x.dat');
hw1y = load('hw1y.dat');

% add a column vector of 1s to the inputs
hw1x = [hw1x, ones(size(hw1x, 1), 1)];

hw1full= [hw1x, hw1y];
hw1full = sortrows(hw1full, 3);


% The number of models
num_models = 10;


% Record training & testing errors for every models
TrainingErrors = zeros(1, num_models);
TestingErrors = zeros(1, num_models);


for ii = 1: 10
    test_index = zeros(100, 1);
    test_index( (ii-1)*10 + 1 : ii * 10 ) = 1;
    
    training_index = ones(100, 1) - test_index;
    test_index = logical(test_index);
    training_index = logical(training_index);
    
    testingX = hw1full(test_index, 1:2);
    testingY = hw1full(test_index, 3);
    trainingX = hw1full(training_index, 1:2);
    trainingY = hw1full(training_index, 3);
    
    for model = 1: num_models
        % train to find model
        [weight_1, x_augument ] = PolyRegress( trainingX, trainingY, model);
        J_TrainError = evaluteTrainError(x_augument, trainingY, weight_1);
        J_TestError = evaluteTrainError(testingX, testingY, weight_1);
   
        % accumulation for average
        TrainingErrors(model) = TrainingErrors(model) + J_TrainError;
        TestingErrors(model) = TestingErrors(model) + J_TestError;
    end
    % plot 
%     figure 
%     plot(testingX(:, end - 1), testingY, 'r.', 'MarkerSize', 10);
%     hold on,
%     plot(trainingX(:, end - 1), trainingY, 'b.', 'MarkerSize', 10);
%     hold off
end

% Average errors on trainging and testing set
TrainingErrors = TrainingErrors ./ 5
TestingErrors = TestingErrors ./ 5


%% From the errors on testing set, we can observe that
%  4 is the best degree for polynomial regression. 
% plot the data and the polynomial obtained.
% get the order-6 fit
[weight_best, hw1x_best] = PolyRegress( hw1x, hw1y, 4 );
error = evaluteTrainError(hw1x, hw1y, weight_best )

% plot the data
figure
plot(hw1x(:, 1), hw1y, 'r.', 'MarkerSize', 10);
% plot the best polynomial fit
hold on
[w_best_plot, hw1x_best_plot] = PolyRegress( linspace(min(hw1x(:, 1)), ...
    max(hw1x(:, 1)), 100)', zeros(100, 1), 4);
plot(hw1x_best_plot(:, end - 1), hw1x_best_plot*weight_best, 'b' );
legend('Training data', 'Order-4 fit');
title('1-Cross Validation on ordered data');
hold off;