function [ error ] = RMSE( X, y, weight )
%RMSE 此处显示有关此函数的摘要
%   此处显示详细说明

% compute predictions
predictions = X * weight';

% obtain the Root Mean Squared Error
error = sqrt( mean((y - predictions).^2) );

end

