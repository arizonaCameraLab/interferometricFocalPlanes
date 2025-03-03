function G = MeasureEdge(d, theta, I, a, alpha1, alpha2)
% d: sensor pair center distance to edge at origin
% theta: diagonal sensor pair angle wrt to x-axis defined CCW positive
% I: Intensity of the edge
% a: distance between each sensor point to their center location
% alpha1: Operator H's angle for diagonal sensor pair
% alpha2: Operator H's angle for anti-diagonal sensor pair

% Here the psf is assumend to be gaussian function
% edge is located at the y axis

% Convert alpha1 and alpha2 to column vectors to ensure correct matrix operations
alpha1 = alpha1(:);
alpha2 = alpha2(:);

% Define 2 pair of sensor coordinates:diagonal and anti-diagonal
x1 = d - a * cos(theta); x2 = d + a * cos(theta);
y1 = -a * sin(theta); y2 = a * sin(theta);
x3 = d - a * cos(pi/2 - theta); x4 = d + a * cos(pi/2 - theta);
y3 = a * sin(pi/2 - theta); y4 = -a * sin(pi/2 - theta);

% Calculate the mutual intensity elements of J
% A12: diagonal sensor pair first measurments: Intensity
% B12: diagonal sensor pair second measurments: Intensity
% C12: diagonal sensor mutual intensity 
A12 = sqrt(pi)/2 * (1+erf(x1));
B12 = sqrt(pi)/2 * (1+erf(x2));
C12 = sqrt(pi)/2 * exp(-(y1-y2).^2/4 - (x1-x2).^2/4) .* (1+erf((x1+x2)/2));

% A12: anti-diagonal sensor pair first measurments: Intensity
% B12: anti-diagonal sensor pair second measurments: Intensity
% C12: anti-diagonal sensor mutual intensity 
A34 = sqrt(pi)/2 * (1+erf(x3));
B34 = sqrt(pi)/2 * (1+erf(x4));
C34 = sqrt(pi)/2 * exp(-(y3-y4).^2/4 - (x3-x4).^2/4) .* (1+erf((x3+x4)/2));

% G: measurments after operator H on to J
    % since g = diag[HJH*], g is 2 by 1 here
% G1, G2: diagonal sensor measurments
% G3, G4: diagonal sensor measurments
% Compute the intensity measurements at different angles
G1 = I*(A12*cos(alpha1).^2 + B12*sin(alpha1).^2 - C12*sin(2*alpha1));
G2 = I*(B12*cos(alpha1).^2 + A12*sin(alpha1).^2 + C12*sin(2*alpha1));
G3 = I*(A34*cos(alpha2).^2 + B34*sin(alpha2).^2 - C34*sin(2*alpha2));
G4 = I*(B34*cos(alpha2).^2 + A34*sin(alpha2).^2 + C34*sin(2*alpha2));

% Combine results into a single matrix output
G = [G1; G2; G3; G4];

end
