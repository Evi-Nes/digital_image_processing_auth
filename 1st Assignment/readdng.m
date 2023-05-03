function [rawim, XYZ2Cam, wbcoeffs ] = readdng(filename)
%% Load Image
obj = Tiff(filename ,'r');
offsets = getTag(obj ,'SubIFD');
setSubDirectory(obj , offsets (1));
rawim = read(obj);

%% Load Metadata
meta_info = imfinfo(filename);

% (x_origin , y_origin ) is the uper left corner of the useful part of the
% sensor and consequently of the array rawim
y_origin = meta_info.SubIFDs{1}.ActiveArea(1) +1;
x_origin = meta_info.SubIFDs{1}.ActiveArea(2) +1;

% width and height of the image (the useful part of array rawim )
width = meta_info.SubIFDs{1}.DefaultCropSize(1);
height = meta_info.SubIFDs{1}.DefaultCropSize(2);

% sensor values corresponding to black and white
blacklevel = meta_info.SubIFDs{1}.BlackLevel(1); 
whitelevel = meta_info.SubIFDs{1}.WhiteLevel; 

wbcoeffs = (meta_info.AsShotNeutral).^(-1);
% green channel will be left unchanged
wbcoeffs = wbcoeffs / wbcoeffs(2); 

XYZ2Cam = meta_info.ColorMatrix2 ;
XYZ2Cam = reshape (XYZ2Cam ,3 ,3)';

% resize array to fit within bounds described by metadata.
rawim = rawim(y_origin:y_origin+height-1,x_origin:x_origin+width-1); 
% transform array to doubles to keep floating point precision.
rawim = double(rawim); 

% scale values to dynamic range of image & cut values beyond the max white level.
rawim = (rawim - blacklevel) ./ (whitelevel - blacklevel); 
rawim = max(0,min(rawim,1)); 

close(obj);
end