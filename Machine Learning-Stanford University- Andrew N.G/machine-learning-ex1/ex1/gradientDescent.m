function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
%J1=0;
%J2=0;
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
%for i=1:m
 %   h = theta(1,1)*X(i,1) + theta(2,1)*X(i,2);% h(x)
   % H1= (h- y(i,1))*X(i,1);
    %H2=(h-y(i,1))*X(i,2);
    %J%1= J1 + H1 ;
    %J2%=J2+H2;
  
%end   
h=X*theta;  % 97*1
diff = h-y;  %97*1
prod=X' * diff;    % 2*1
prod = (alpha/m)*prod;
theta=theta - prod;





%theta(1,1)=theta(1,1) - (alpha/m)*J1;
%theta(2,1)=theta(2,1) - (alpha/m)*J2;
 % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
   
end

