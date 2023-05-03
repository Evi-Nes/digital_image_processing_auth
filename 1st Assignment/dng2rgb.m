function [Csrgb, Clinear, Cxyz, Ccam, RGBsaturated] = dng2rgb(rawim, XYZ2Cam, wbcoeffs, bayertype, method)
% Color Space Matrix
XYZ2RGB = [[+3.2406, -1.5372, -0.4986]; [-0.9689, +1.8758, +0.0415]; [+0.0557, -0.2040, +1.0570]];

%% White Ballance
mask = wbmask(size(rawim,1), size(rawim,2), wbcoeffs, bayertype);
balanced_img = rawim .* mask;

%% Color Space Transformstions
temp = uint16(balanced_img * (2^16 - 1));
demosaiced_img = cdemosaic(temp, bayertype, method);

Clinear = double(demosaiced_img) / (2^16 - 1);

Csrgb = Clinear .^ (1/2.2);
Csrgb = max(0, min(Csrgb,1)); 
Cxyz = apply_cmatrix(Clinear, XYZ2RGB);
Ccam = apply_cmatrix(Clinear, XYZ2Cam);

% Increase saturation by 50%
HSV = rgb2hsv(Csrgb);
HSV(:, :, 2) = HSV(:, :, 2) * 1.5;

RGBsaturated = hsv2rgb(HSV);

end