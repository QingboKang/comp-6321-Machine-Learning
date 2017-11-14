% This is the solution for the NO.1 problem of Assignment 1
% Linear and polynomial regression
% Author: Qingbo Kang 
% Student ID: 40058122
% e-mail: qi_kang@encs.concordia.ca

close all;
%% a
% load the data into memory and plot it
hw1x = load('hw1x.dat');
hw1y = load('hw1y.dat');

% plot data
plot(hw1x, hw1y, 'r.', 'MarkerSize', 10);
legend('Training data');
title('1-a');

%% b
% add a column vector of 1s to the inputs
hw1x = [hw1x, ones(size(hw1x, 1), 1)];

% obtain the weight vector w use the linear regression formula
w = (hw1x'*hw1x)^-1 * hw1x' * hw1y;

% plot the linear regression line with the data 
figure
plot(hw1x(:, end - 1), hw1y, 'r.', 'MarkerSize', 10);
hold on
plot(hw1x(:, end - 1), hw1x*w, 'b-' );
legend('Training data', 'Linear regression');
title('1-b');
hold off

%% c
training_error = evaluteTrainError( hw1x, hw1y, w );

% report the error
fprintf( 'The training error of the resulting fit is: %f\n', training_error);

%% d, e
% get a quadratic fit of the data
[w_quad, hw1x_quad]  = PolyRegress( hw1x, hw1y, 2 );

% plot the data
figure
plot(hw1x(:, 1), hw1y, 'r.', 'MarkerSize', 10);
% plot the quadratic fit
hold on
[w_quad_plot, hw1x_cubic_plot] = PolyRegress( linspace(-5, 12, 100)',...
    zeros(100, 1),2);

plot(hw1x_cubic_plot(:, end - 1), hw1x_cubic_plot*w_quad, 'b' );
legend('Training data', 'Quadratic fit');
title('1-e');
hold off;

% report the training error
training_error_quad = evaluteTrainError( hw1x_quad, hw1y, w_quad );
fprintf( 'The training error of the quadratic fit is: %f\n', ...
    training_error_quad);

%% f
% get a cubic fit of the data
[w_cubic, hw1x_cubic] = PolyRegress( hw1x, hw1y, 3 );

% plot the data
figure
plot(hw1x(:, 1), hw1y, 'r.', 'MarkerSize', 10);
% plot the quadratic fit
hold on
[w_cubic_plot, hw1x_cubic_plot] = PolyRegress( linspace(-5, 12, 100)',...
    zeros(100, 1), 3);
plot(hw1x_cubic_plot(:, end - 1), hw1x_cubic_plot*w_cubic, 'b' );
legend('Training data', 'Cubic fit');
title('1-f');
hold off;

% report the training error
training_error_quad = evaluteTrainError( hw1x_cubic, hw1y, w_cubic );
fprintf( 'The training error of the cubic fit is: %f\n', ...
    training_error_quad);

%% g

%% h
FiveFoldCrossValidation;

%% i
% normalize the input data in each column by the maximum absolute vale in 
% that column.
hw1x = [hw1x(:, 1)./max(abs(hw1x(:, 1))), hw1x(:, 2)];

% performs five-fold cross-validation 
% FiveFoldCrossValidation;