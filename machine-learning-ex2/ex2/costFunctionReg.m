function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

probabilities = sigmoid(X * theta);

ones = find(y==1);
zeros = find(y==0);

rows_theta = size(theta)(1);

theta_dis = theta((2:rows_theta),:);

J = (sum(log(probabilities(ones))) + sum(log(1-probabilities(zeros)))) * (-1/(m)) + (theta_dis'*theta_dis)*(lambda/(2*m));

errors = probabilities - y;



grad = (sum(errors.*X) * 1/m)' + [0;(lambda/m)*theta_dis];




% =============================================================

end
