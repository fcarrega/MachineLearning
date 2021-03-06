function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Add bias unit to X
X = [ones(m, 1) X];

% Compute a2
a2 = sigmoid(Theta1 * X')';

% Add bias unit to a2
a2 = [ones(size(a2, 1), 1) a2];

% Compute h
h = sigmoid(Theta2 * a2')';

% For each training example
for i = 1:size(X,1)

  % Build y vector
  vec_y = zeros(num_labels, 1);
  vec_y(y(i)) = 1;
  
  % For each class
  for k = 1:num_labels
    % Sum cost with existing cost
    J = J + (- vec_y(k) * log(h(i, k))) - (1-vec_y(k)) * log (1 - h(i, k));
  end
end

% Divide total cost by number of training examples
J = J / m;

% Initialize regularization term
reg_term = 0;

% Remove biases from theta matrices
temp_Theta1 = Theta1(:,2:end);
temp_Theta2 = Theta2(:,2:end);

% Compute regularization term
reg_term = sum(sum(temp_Theta1 .^ 2)) + sum(sum(temp_Theta2 .^ 2));

% Final computation of regularization term
reg_term = reg_term * lambda / (2*m);

% Add regularization term to cost
J = J + reg_term;

% Backpropagation

% Loop on each training example
for t = 1:m

  % Step 1 - Compute feedforward
  
  % Initialize layer 1 with X (including bias)
  a_1 = X(t, :);
  
  % Compute z_2
  z_2 = Theta1 * a_1';
  
  % Compute a_2
  a_2 = [1; sigmoid(z_2)];
  
  % Compute z_3
  z_3 = Theta2 * a_2;
  
  % Compute a_3
  a_3 = sigmoid(z_3);
  
  % Step 2 - Compute errors for last layer
  delta_3 = zeros(num_labels, 1);

  % Build y vector
  vec_y = zeros(num_labels, 1);
  vec_y(y(t)) = 1;  

  delta_3 = a_3 - vec_y;
  
  % Step 2 - Compute error for hidden layer
  delta_2 = (Theta2' * delta_3) .* (a_2.*(1-a_2));
  delta_2 = delta_2(2:end);
  
  % Accumulate gradients
  Theta1_grad = Theta1_grad + delta_2 * a_1;
  Theta2_grad = Theta2_grad + delta_3 * a_2';
  
end

Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

% Add regulatization term except on bias term (fist column)
Theta1_grad = [Theta1_grad(:,1) Theta1_grad(:,2:end) + (lambda / m) * Theta1(:,2:end)];
Theta2_grad = [Theta2_grad(:,1) Theta2_grad(:,2:end) + (lambda / m) * Theta2(:,2:end)];


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
