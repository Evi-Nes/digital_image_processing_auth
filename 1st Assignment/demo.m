%% Load and Read .DNG image
filename = "RawImage.DNG";
[rawim, XYZ2Cam, wbcoeffs] = readdng (filename);

%% Transform to RGB
% The variables for the demosaic
bayertype = 'rggb';
method = 'linear';

[Csrgb, Clinear, Cxyz, Ccam, RGBsaturated] = dng2rgb(rawim, XYZ2Cam, wbcoeffs, bayertype, method);

%% Write image files from the dng2rgb output
if strcmp(method, 'linear')
    imwrite(Csrgb, 'lin'+"_"+bayertype+"_"+"RGB.jpg");
    imwrite(Clinear, 'lin'+"_"+bayertype+"_"+"linear.jpg");
    imwrite(Cxyz, 'lin'+"_"+bayertype+"_"+"xyz.jpg");
    imwrite(Ccam, 'lin'+"_"+bayertype+"_"+"cam.jpg");
    imwrite(RGBsaturated, 'lin'+"_"+bayertype+"_"+"RGBsaturated.jpg");
else
    imwrite(Csrgb, 'near'+"_"+bayertype+"_"+"RGB.jpg");
    imwrite(Clinear, 'near'+"_"+bayertype+"_"+"linear.jpg");
    imwrite(Cxyz, 'near'+"_"+bayertype+"_"+"xyz.jpg");
    imwrite(Ccam, 'near'+"_"+bayertype+"_"+"cam.jpg");
    imwrite(RGBsaturated, 'near'+"_"+bayertype+"_"+"RGBsaturated.jpg"); 
end

%% Plot the output from the dng2rgb function
% nbins = 70;
% 
% red = Csrgb(:,:,1);
% green = Csrgb(:,:,2);
% blue = Csrgb(:,:,3);
% 
% figure(1)
% h1 = histogram(red,nbins,'FaceColor','r');
% hold on
% h2 = histogram(green,nbins,'FaceColor','g');
% h3 = histogram(blue,nbins,'FaceColor','b');
% title('Histogram of Csrgb')
% hold off
% 
% red = RGBsaturated(:,:,1);
% green = RGBsaturated(:,:,2);
% blue = RGBsaturated(:,:,3);
% 
% figure(2)
% h1 = histogram(red,nbins,'FaceColor','r');
% hold on
% h2 = histogram(green,nbins,'FaceColor','g');
% h3 = histogram(blue,nbins,'FaceColor','b');
% title('Histogram of RGBsaturated')
% hold off
% 
% red = Cxyz(:,:,1);
% green = Cxyz(:,:,2);
% blue = Cxyz(:,:,3);
% 
% figure(3)
% h1 = histogram(red,nbins,'FaceColor','r');
% hold on
% h2 = histogram(green,nbins,'FaceColor','g');
% h3 = histogram(blue,nbins,'FaceColor','b');
% title('Histogram of Cxyz')
% hold off
% 
% red = Clinear(:,:,1);
% green = Clinear(:,:,2);
% blue = Clinear(:,:,3);
% 
% figure(4)
% h1 = histogram(red,nbins,'FaceColor','r');
% hold on
% h2 = histogram(green,nbins,'FaceColor','g');
% h3 = histogram(blue,nbins,'FaceColor','b');
% title('Histogram of Clinear')
% hold off
% 
% red = Ccam(:,:,1);
% green = Ccam(:,:,2);
% blue = Ccam(:,:,3);
% 
% figure(5)
% h1 = histogram(red,nbins,'FaceColor','r');
% hold on
% h2 = histogram(green,nbins,'FaceColor','g');
% h3 = histogram(blue,nbins,'FaceColor','b');
% title('Histogram of Ccam')
% hold off