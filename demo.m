clc 
clf

%% Load and Read .DNG image
filename = "RawImage.DNG";

[rawim, XYZ2Cam, wbcoeffs] = readdng (filename);

%% Transform to rgb
bayertype = 'rggb';

% ask the user to choose method for interpolation.
% promt = 'Choose method: 1 for nearest, 2 for linear';
% x = input(promt);
% if x == 1
%     method = 'nearest';
% else
%     method = 'linear';
% end
method = 'linear';
[Csrgb, Clinear, Cxyz, Ccam] = dng2rgb(rawim, XYZ2Cam, wbcoeffs, bayertype, method);

% Write image files from the dng2rgb output
imwrite(Csrgb, method+"_"+bayertype+"_"+"rgb.jpg");
imwrite(Clinear, method+"_"+bayertype+"_"+"linear.jpg");
imwrite(Cxyz, method+"_"+bayertype+"_"+"xyz.jpg");
imwrite(Ccam, method+"_"+bayertype+"_"+"cam.jpg");