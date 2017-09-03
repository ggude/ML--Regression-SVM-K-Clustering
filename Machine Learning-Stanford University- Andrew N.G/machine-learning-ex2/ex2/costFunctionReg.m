function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
[p,q] = size( X );
% You need to return the following variables correctly 
J = 0;
theta1=0;
grad = zeros(size(theta));
h = sigmoid(X*theta);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
  %J = J + [-y(1,1)*log(h(1,1)) - (1-y(1,1))*log(1-h(1,1))];
for i = 1:m
  %h = theta(1,1)*X(i,1) + theta(2,1)*X(i,2) + theta(3,1)*X(i,3);
  J = J + [-y(i,1)*log(h(i,1)) - (1-y(i,1))*log(1-h(i,1))];
end
for j = 2:q
  theta1 = theta1 + theta(j,1)^2;
end
t=lambda/(2*m);
 theta1 = t*theta1;
 J = J/m;
 J = J + theta1; 
 %gradient
diff = h-y;
prod = X'*diff;
grad1 = prod/m;  % eq (1)--
k = lambda/m;
grad(1,1) = grad1(1,1);
for l = 2:q
grad(l,1) = grad1(l,1) + k*theta(l,1);    %eq (2)
end

% =============================================================

