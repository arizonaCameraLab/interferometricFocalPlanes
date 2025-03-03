clear; clc;
syms x1 y1 x2 y2 I1 I2 a real

% Define PSF function
psf = @(x, y) exp(-(x.^2 + y.^2) / 2);
% psf = @(x, y) sinc(x) .* sinc(y);

% use theta to control the orientation of the second sensor array
theta = -pi/4;
sx = sqrt(2)*a*cos(theta);
sy = sqrt(2)*a*sin(theta);

% Define functions A, B, C
A1 = I1 * psf(x1 - a, y1 - a).^2 + I2 * psf(x2 - a, y2 - a).^2;
B1 = I1 * psf(x1 + a, y1 + a).^2 + I2 * psf(x2 + a, y2 + a).^2;
C1 = I1 * psf(x1 + a, y1 + a) .* psf(x1 - a, y1 - a) + I2 * psf(x2 + a, y2 + a) .* psf(x2 - a, y2 - a);

A2 = I1 * psf(x1 - sx, y1 - sy).^2 + I2 * psf(x2 - sx, y2 - sy).^2;
B2 = I1 * psf(x1 + sx, y1 + sy).^2 + I2 * psf(x2 + sx, y2 + sy).^2;
C2 = I1 * psf(x1 - sx, y1 - sy) .* psf(x1 + sx, y1 + sy) + I2 * psf(x2 - sx, y2 - sy) .* psf(x2 + sx, y2 + sy);

% Define function vector
F = [A1; B1; C1; A2; B2; C2];

% Compute Jacobian matrix J_F = dF/d[x1, y1, x2, y2, I1, I2]
vars = [x1, y1, x2, y2, I1, I2];
J_F = jacobian(F, vars);

% Convert Jacobian matrix to numerical function for faster evaluation
J_F_func = matlabFunction(J_F, 'Vars', {x1, y1, x2, y2, I1, I2, a});

% Display symbolic Jacobian matrix
disp('Jacobian Matrix:');
disp(J_F);

% Example numerical substitution
x1_val = 0.51;
y1_val = -0.31;
x2_val = 0.71;
y2_val = 0.21;
I1_val = 2.1;
I2_val = 1.6;
a_val = 0.55;

% Evaluate Jacobian numerically
J_num = double(J_F_func(x1_val, y1_val, x2_val, y2_val, I1_val, I2_val, a_val));

% Display numerical Jacobian
disp('Numerical Jacobian:');
disp(J_num);
disp(rank(J_num))
disp(cond(J_num))

% the rank of J will be 6 if use sinc psf, and theta~=-pi/4, by breaking
% the symmetry
