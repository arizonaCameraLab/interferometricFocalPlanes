function [G1_func, G2_func, G3_func, G4_func] = MeasureLine()
    syms d theta I a alpha1 alpha2 real;
    
    % Measurement Functions
    x1 = d - a * cos(theta); x2 = d + a * cos(theta);
    y1 = -a * sin(theta); y2 = a * sin(theta);
    x3 = d - a * cos(pi/2 - theta); x4 = d + a * cos(pi/2 - theta);
    y3 = -a * sin(pi/2 - theta); y4 = a * sin(pi/2 - theta);
    psf = @(x) exp(-x.^2/2);
    % psf = @(x) sinc(x);
    
    A1 = psf(x1)^2; B1 = psf(x2)^2; C1 = psf(x1)*psf(x2)*psf((y1 - y2)/sqrt(2));
    A2 = psf(x3)^2; B2 = psf(x4)^2; C2 = psf(x3)*psf(x4)*psf((y3 - y4)/sqrt(2));
    
    G1 = I*(A1*cos(alpha1)^2 + B1*sin(alpha1)^2 - C1*sin(2*alpha1));
    G2 = I*(B1*cos(alpha1)^2 + A1*sin(alpha1)^2 + C1*sin(2*alpha1));
    G3 = I*(A2*cos(alpha2)^2 + B2*sin(alpha2)^2 - C2*sin(2*alpha2));
    G4 = I*(B2*cos(alpha2)^2 + A2*sin(alpha2)^2 + C2*sin(2*alpha2));
    
    % MATLAB Functions
    G1_func = matlabFunction(G1, 'Vars', [d, theta, I, a, alpha1]);
    G2_func = matlabFunction(G2, 'Vars', [d, theta, I, a, alpha1]);
    G3_func = matlabFunction(G3, 'Vars', [d, theta, I, a, alpha2]);
    G4_func = matlabFunction(G4, 'Vars', [d, theta, I, a, alpha2]);

end