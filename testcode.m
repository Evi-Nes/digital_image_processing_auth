% Load the raw image
filename = "RawImage.DNG";
[raw_im, XYZ2Cam, wbcoeffs] = readdng (filename);

tiff_info = imfinfo(filename);

% Set the size of the image
rows = tiff_info.Height;
cols = tiff_info.Width;

% Define the Bayer filter pattern
Bayer = [2 1; 1 3];

% Initialize output image
rgb_img = zeros(rows,cols,3);
bayertype = 'rggb';

mask = wbmask(size(raw_im,1), size(raw_im,2), wbcoeffs, bayertype);
balancedim = raw_im .* mask;

% Color Space Transformstions
raw_img = uint16(balancedim * (2^16 - 1));

% Apply Bayer demosaic
for i = 2:rows-1
    for j = 2:cols-1
        
        % Compute the missing color values
        if (mod(i,2) == 0 && mod(j, 2) == 0)
            % Blue pixel
            blue_val = raw_img(i,j);
            green_val = 0.25 * (raw_img(i-1,j) + raw_img(i+1,j) + raw_img(i,j-1) + raw_img(i,j+1));
            red_val = 0.25 * (raw_img(i-1,j-1) + raw_img(i+1,j+1) + raw_img(i-1,j+1) + raw_img(i+1,j-1));

        elseif (mod(i,2) == 0 || mod(j, 2) == 0)
            % Green pixel
            red_val = 0.5 * (raw_img(i,j-1) + raw_img(i,j+1));
            green_val = raw_img(i,j);
            blue_val = 0.5 * (raw_img(i-1,j) + raw_img(i+1,j));

        else
            % Red pixel
            red_val = raw_img(i,j);
            green_val = 0.25 * (raw_img(i-1,j) + raw_img(i,j-1) + raw_img(i,j+1) + raw_img(i+1,j));
            blue_val = 0.25 * (raw_img(i-1,j-1) + raw_img(i-1,j+1) + raw_img(i+1,j-1) + raw_img(i+1,j+1));
        end

        % Store the color values in the output image
        rgb_img(i,j,1) = red_val;
        rgb_img(i,j,2) = green_val;
        rgb_img(i,j,3) = blue_val;
    end
end

% Display the output image
imshow(rgb_img);

XYZ2RGB = [[+3.2406, -1.5372, -0.4986]; [-0.9689, +1.8758, +0.0415]; [+0.0557, -0.2040, +1.0570]];

Clinear = double(rgb_img) / (2^16 - 1);

Csrgb = Clinear .^ (1/2.2);
Csrgb = max(0, min(Csrgb,1)); 
imshow(Csrgb);
imshow(Clinear);
