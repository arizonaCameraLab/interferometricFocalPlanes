
function G = MeasureCorner(x0, y0, theta, I, a, alpha1, alpha2)

alpha1 = alpha1(:);  % alpha1 to column vector
alpha2 = alpha2(:);  % alpha2 to column vector

x1 = x0 - a * cos(theta); x2 = x0 + a * cos(theta);
y1 = y0 - a * sin(theta); y2 = y0 + a * sin(theta);
x3 = x0 - a * cos(pi/2 - theta); x4 = x0 + a * cos(pi/2 - theta);
y3 = y0 + a * sin(pi/2 - theta); y4 = y0 - a * sin(pi/2 - theta);

A12 = pi/4 * (1+erf(x1)) * (1+erf(y1));
B12 = pi/4 * (1+erf(x2)) * (1+erf(y2));
C12 = pi/4 * exp(-(y1-y2).^2/4 - (x1-x2).^2/4) .* ...
    (1+erf((x1+x2)/2)) .* (1+erf((y1+y2)/2));

A34 = pi/4 * (1+erf(x3)) * (1+erf(y3));
B34 = pi/4 * (1+erf(x4)) * (1+erf(y4));
C34 = pi/4 * exp(-(y3-y4).^2/4 - (x3-x4).^2/4) .* ...
    (1+erf((x3+x4)/2)) .* (1+erf((y3+y4)/2));

G1 = I*(A12*cos(alpha1).^2 + B12*sin(alpha1).^2 - C12*sin(2*alpha1));
G2 = I*(B12*cos(alpha1).^2 + A12*sin(alpha1).^2 + C12*sin(2*alpha1));
G3 = I*(A34*cos(alpha2).^2 + B34*sin(alpha2).^2 - C34*sin(2*alpha2));
G4 = I*(B34*cos(alpha2).^2 + A34*sin(alpha2).^2 + C34*sin(2*alpha2));

G = [G1; G2; G3; G4];

end