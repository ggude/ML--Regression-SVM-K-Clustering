function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.
%data = load('ex2data1.txt');
% You need to return the following variables correctly 
%g = zeros(size(z));
%X=data(:, [1, 2]);
%[m, n] = size(X);
%X = [ones(m, 1) X];
%initial_theta = zeros(n + 1, 1);
[p,q]=size( z );
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
for i = 1:p
    for j = 1:q

        g(i,j) = 1/(1+exp(-z(i,j)));

    end 
end



% =============================================================


