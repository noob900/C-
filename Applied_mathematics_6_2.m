f = @(x0) [x0(1)^3 - x0(2)^2 - 2; x0(1)*x0(2) - 5*x0(2)  + 10]; % write your system of equations here
x0 = [0;0];
h = sqrt(eps);
J = numJacobian(f,x0,h);

function J = numJacobian(f,x0,h)
 % Calculate each partial derivative manually (your style)

    % Partial derivatives for first equation (f1) with respect to x1 and x2
    J11 = (f([x0(1) + h; x0(2)]) - f(x0)) / h;
    J12 = (f([x0(1); x0(2) + h]) - f(x0)) / h;

    % Partial derivatives for second equation (f2) with respect to x1 and x2
    J21 = (f([x0(1) + h; x0(2)]) - f(x0)) / h;
    J22 = (f([x0(1); x0(2) + h]) - f(x0)) / h;

    % Construct the Jacobian matrix
    J = [J11(1), J12(1);  % First equation's partials
         J21(2), J22(2)];  % Second equation's partials

    % Display the Jacobian matrix
    disp('Jacobian matrix:');
    disp(J);

end 

% Define the system of equations as a function handle
