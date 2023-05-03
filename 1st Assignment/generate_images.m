filename = "RawImage.DNG";
[rawim, XYZ2Cam, wbcoeffs] = readdng(filename);

bayertypes = ["bggr", "gbrg", "grbg", "rggb"];
methods = ["linear", "nearest"];

for i = 1 : 4
    for j = 1 : 2
        bayertype = bayertypes(i);
        method = methods(j);

        [Csrgb, Clinear, Cxyz, Ccam, RGBsaturated] = dng2rgb(rawim, XYZ2Cam, wbcoeffs, bayertype, method);

        cd(method+"_"+bayertype);
        % Write image files from the dng2rgb output
        imwrite(Csrgb, method+"_"+bayertype+"_"+"rgb.jpg");
        imwrite(Clinear, method+"_"+bayertype+"_"+"linear.jpg");
        imwrite(Cxyz, method+"_"+bayertype+"_"+"xyz.jpg");
        imwrite(Ccam, method+"_"+bayertype+"_"+"cam.jpg");
        imwrite(RGBsaturated, method+"_"+bayertype+"_"+"cam.jpg");

% Plot the output from the dng2rgb function
nbins = 70;

red = Csrgb(:,:,1);
green = Csrgb(:,:,2);
blue = Csrgb(:,:,3);

figure(1)
h1 = histogram(red,nbins,'FaceColor','r');
hold on
h2 = histogram(green,nbins,'FaceColor','g');
h3 = histogram(blue,nbins,'FaceColor','b');
title('Histogram of Csrgb')
hold off
saveas(gcf, method+"_"+bayertype+"_Csrgb_histogram.jpg");

red = RGBsaturated(:,:,1);
green = RGBsaturated(:,:,2);
blue = RGBsaturated(:,:,3);

figure(2)
h1 = histogram(red,nbins,'FaceColor','r');
hold on
h2 = histogram(green,nbins,'FaceColor','g');
h3 = histogram(blue,nbins,'FaceColor','b');
title('Histogram of RGBsaturated')
hold off
saveas(gcf, method+"_"+bayertype+"_RGBsatur_histogram.jpg");

red = Cxyz(:,:,1);
green = Cxyz(:,:,2);
blue = Cxyz(:,:,3);

figure(3)
h1 = histogram(red,nbins,'FaceColor','r');
hold on
h2 = histogram(green,nbins,'FaceColor','g');
h3 = histogram(blue,nbins,'FaceColor','b');
title('Histogram of Cxyz')
hold off
saveas(gcf, method+"_"+bayertype+"_Cxyz_histogram.jpg");

red = Clinear(:,:,1);
green = Clinear(:,:,2);
blue = Clinear(:,:,3);

figure(4)
h1 = histogram(red,nbins,'FaceColor','r');
hold on
h2 = histogram(green,nbins,'FaceColor','g');
h3 = histogram(blue,nbins,'FaceColor','b');
title('Histogram of Clinear')
hold off
saveas(gcf, method+"_"+bayertype+"_Clinear_histogram.jpg");

red = Ccam(:,:,1);
green = Ccam(:,:,2);
blue = Ccam(:,:,3);

figure(5)
h1 = histogram(red,nbins,'FaceColor','r');
hold on
h2 = histogram(green,nbins,'FaceColor','g');
h3 = histogram(blue,nbins,'FaceColor','b');
title('Histogram of Ccam')
hold off

saveas(gcf, method+"_"+bayertype+"_cam_histogram.jpg");
cd("..");
    end
end
