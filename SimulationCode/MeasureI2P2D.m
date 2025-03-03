function G = MeasureI2P2D(x1, y1, x2, y2, I1, I2, a, alpha1, alpha2)

alpha1 = alpha1(:);
alpha2 = alpha2(:);

% psf = @(x, y) sinc(x) .* sinc(y);         % for sinc psf
psf = @(x, y) exp(-(x.^2 + y.^2) / 2);      % for Gaussian PSF

A1 = I1*psf(x1 - a, y1 - a).^2 + I2*psf(x2 - a, y2 - a).^2;
B1 = I1*psf(x1 + a, y1 + a).^2 + I2*psf(x2 + a, y2 + a).^2;
C1 = I1*psf(x1 + a, y1 + a) .* psf(x1 - a, y1 - a) + I2*psf(x2 + a, y2 + a) .* psf(x2 - a, y2 - a);

A2 = I1*psf(x1 - a, y1 + a).^2 + I2*psf(x2 - a, y2 + a).^2;
B2 = I1*psf(x1 + a, y1 - a).^2 + I2*psf(x2 + a, y2 - a).^2;
C2 = I1*psf(x1 + a, y1 - a) .* psf(x1 - a, y1 + a) + I2*psf(x2 + a, y2 - a) .* psf(x2 - a, y2 + a);

G1 = (A1 .* cos(alpha1).^2 + B1 .* sin(alpha1).^2 - C1 .* sin(2 * alpha1));
G2 = (B1 .* cos(alpha1).^2 + A1 .* sin(alpha1).^2 + C1 .* sin(2 * alpha1));
G3 = (A2 .* cos(alpha2).^2 + B2 .* sin(alpha2).^2 - C2 .* sin(2 * alpha2));
G4 = (B2 .* cos(alpha2).^2 + A2 .* sin(alpha2).^2 + C2 .* sin(2 * alpha2));

G = [G1; G2; G3; G4];

end