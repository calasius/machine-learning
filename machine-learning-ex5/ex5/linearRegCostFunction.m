function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

rows_theta = size(theta)(1);

theta_dis = theta((2:rows_theta),:);

predictions = X*theta;
  
errors = predictions - y;

J = sum(errors.^2) *  1/(2*m) + (theta_dis'*theta_dis)*(lambda/(2*m));

grad = sum(errors.*X) * 1/m ;

grad = grad' + [0;(lambda/m)*theta_dis];











% =========================================================================

grad = grad(:);

end
