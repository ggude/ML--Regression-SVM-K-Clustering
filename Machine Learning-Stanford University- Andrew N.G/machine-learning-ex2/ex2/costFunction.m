function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for the logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
%g=sigmoid( z );
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
h = sigmoid(X*theta);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

for i = 1:m
 %h = theta(1,1)*X(i,1) + theta(2,1)*X(i,2) + theta(3,1)*X(i,3);
 J = J + [-y(i,1)*log(h(i,1)) - (1-y(i,1))*log(1-h(i,1))];
 end 
J = J/m;
diff = h-y;
prod = X'*diff;
grad = prod/m;



% =============================================================


