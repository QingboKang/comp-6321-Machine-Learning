function [ error ] = RMSE( X, y, weight )
%RMSE �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��

% compute predictions
predictions = X * weight';

% obtain the Root Mean Squared Error
error = sqrt( mean((y - predictions).^2) );

end

