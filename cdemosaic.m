function [rgb_img] = cdemosaic(raw_img, bayertype, method)
    % Set the size of the image
    [rows, cols] = size(raw_img);
    rgb_img = zeros(rows,cols,3);

    % Apply Bayer demosaic and compute the missing color values
    for i = 3:rows-2
        for j = 3:cols-2
            
            if (mod(i,2) == 0 && mod(j, 2) == 0)
                % Blue pixel
                blue_val = raw_img(i,j);
                green_val = 0.25 * (raw_img(i-1,j) + raw_img(i+1,j) + raw_img(i,j-1) + raw_img(i,j+1));
                red_val = 0.25 * (raw_img(i-1,j-1) + raw_img(i+1,j+1) + raw_img(i-1,j+1) + raw_img(i+1,j-1));
    
            elseif (mod(i,2) == 0 || mod(j, 2) == 0 && mod(i, 2) == 1)
                % Green pixel between reds
                red_val = 0.5 * (raw_img(i-1,j-1) + raw_img(i+1,j+1));
                green_val = raw_img(i,j);
                blue_val = 0.5 * (raw_img(i,j-1) + raw_img(i,j+1));

            elseif (mod(i,2) == 0 || mod(j, 2) == 0 && mod(i, 2) == 0)
                % Green pixel between blues
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

end