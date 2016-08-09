% function that computes the cost function during linear regression
function J = costComputer(X, y, theta, m)
J = sum((X * theta - y).^2) / ( 2 * m);


