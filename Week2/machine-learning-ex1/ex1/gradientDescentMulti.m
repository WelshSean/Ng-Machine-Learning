function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
n = size(X, 2); % number of features
theta_new = zeros(1,length(theta));

J_history = zeros(num_iters, 1);



for iter = 1:num_iters
    
    % Evaluate h
    h = zeros(1,length(y));
    for i = 1:m
        for j = 1:n
            h(i) = h(i) + X(i,j)*theta(j);
        end
    end

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %calculate the sum of h(x) - y

    for jj = 1:n    % theta Index loop
        sum=0;
        for i = 1:m         % sample loop            
            sum = (h(i) - y(i)) * X(i,jj);
        end
        theta(jj) = theta(jj) - sum * alpha/m;
    end


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

