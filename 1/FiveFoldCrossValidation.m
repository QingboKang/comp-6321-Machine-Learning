%% h 
% Write a procedure that performs five-fold cross-validation on your data.
% Use it to determine the best degree for polynomial regression. 
% Show the data that supports your conclusion, and explain how you have come to this conclusion. 
% For the best fit, plot the data and the polynomial obtained.


% Number of data examples
n = size(hw1x, 1);

% Generate cross-validation indices
indices = crossvalind('Kfold', n, 5);

% The number of models
num_models = 16;


% Record training & testing errors for every models
TrainingErrors = zeros(1, num_models);
TestingErrors = zeros(1, num_models);

for ii = 1:5
    % Partition ii fold for testing
    % The rest of other partitions are used for training
    testing = (indices == ii);
    training = ~testing;
    
    testingX = hw1x(testing, :);
    testingY = hw1y(testing, :);
    trainingX = hw1x(training, :);
    trainingY = hw1y(training, :);
    
    %% loop to find the best fit
    % for each fit, we record the error on training and testing set
    for model = 1: num_models
        [weight, x_augument ] = PolyRegress( trainingX, trainingY, model);
        J_TrainError = evaluteTrainError(x_augument, trainingY, weight);
        J_TestError = evaluteTrainError(testingX, testingY, weight);
        % accumulation for average
        TrainingErrors(model) = TrainingErrors(model) + J_TrainError;
        TestingErrors(model) = TestingErrors(model) + J_TestError;
    end
end

% Average errors on trainging and testing set
TrainingErrors = TrainingErrors ./ 5;
TestingErrors = TestingErrors ./ 5;

% plot errors
figure;
plot(1:num_models, TrainingErrors, 'r', 1:num_models, TestingErrors, 'g');
xlabel('order');
ylabel('error');
title('h-Training errors and testing errors');
legend('Training errors', 'Testing errors');

%% 
% From the errors on testing set, we can observe that
% 6 is the best degree for polynomial regression. 
% plot the data and the polynomial obtained.
% get the order-6 fit
[weight_best, hw1x_best] = PolyRegress( hw1x, hw1y, 6 );


% plot the data
figure
plot(hw1x(:, 1), hw1y, 'r.', 'MarkerSize', 10);
% plot the best polynomial fit
hold on
[w_best_plot, hw1x_best_plot] = PolyRegress( linspace(min(hw1x(:, 1)), ...
    max(hw1x(:, 1)), 100)', zeros(100, 1), 6);
plot(hw1x_best_plot(:, end - 1), hw1x_best_plot*weight_best, 'b' );
legend('Training data', 'Order-6 fit');
title('1-Cross Validation');
hold off;


