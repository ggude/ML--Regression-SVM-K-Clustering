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
[p,q] = size( X );
theta1=0;
%h = X*theta;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
h = X*theta;
for i=1:m
    %h = theta(1,1)*X(i,1) + theta(2,1)*X(i,2);
    J=J + (h(i,1) - y(i,1))^2;
end
%h = X*theta;
%J = J + (h - y).^2;
J=J/(2*m);

for j = 2:q
  theta1 = theta1 + theta(j,1)^2;
end
t=lambda/(2*m);
 theta1 = t*theta1;
 J = J + theta1;



diff = h-y;
prod = X'*diff;
grad1 = prod/m;  % eq (1)--
k = lambda/m;
grad(1,1) = grad1(1,1);
for l = 2:q
grad(l,1) = grad1(l,1) + k*theta(l,1);    %eq (2)
end









% =========================================================================

grad = grad(:);

end
