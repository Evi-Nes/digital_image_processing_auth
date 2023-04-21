%% Load and Read .DNG image
clc 
clf
filename = "RawImage.DNG";
[rawim, XYZ2Cam, wbcoeffs] = readdng (filename);

%% Transform to RGB
bayertype = 'rggb';

% ask the user to choose method for interpolation.
promt = 'Choose method: 1 for nearest, 2 for linear ';
x = input(promt);
if x == 1
    method = 'nearest';
else
    method = 'linear';
end

[Csrgb, Clinear, Cxyz, Ccam] = dng2rgb(rawim, XYZ2Cam, wbcoeffs, bayertype, method);

%% Show outputs

% Write image files from the dng2rgb output
imwrite(Csrgb, method+"_"+bayertype+"_"+"RGB.jpg");
% imwrite(Clinear, method+"_"+bayertype+"_"+"linear.jpg");
% imwrite(Cxyz, method+"_"+bayertype+"_"+"xyz.jpg");
% imwrite(Ccam, method+"_"+bayertype+"_"+"cam.jpg");

% Plot the output from the dng2rgb function
red = Csrgb(:,:,1);
green = Csrgb(:,:,2);
blue = Csrgb(:,:,3);
nbins = 50;

figure
h1 = histogram(red,nbins,'FaceColor','r');
hold on
h2 = histogram(green,nbins,'FaceColor','g');
h3 = histogram(blue,nbins,'FaceColor','b');
title('Histogram of Csrgb')
hold off
