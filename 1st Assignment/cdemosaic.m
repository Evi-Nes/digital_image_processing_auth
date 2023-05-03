function [rbg_img] = cdemosaic(raw_img, bayertype, method)
% Function that calculates the demosaiced image depending on the bayertype
% and method provided
    [rows, cols] = size(raw_img);
    rbg_img = zeros(rows,cols,3);

    if method == "linear"
        if bayertype == "bggr"
            rbg_img = linear_bggr_demosaic(raw_img, rows, cols);
        elseif bayertype == "gbrg"
            rbg_img = linear_gbrg_demosaic(raw_img, rows, cols);
        elseif bayertype == "grbg"
            rbg_img = linear_grbg_demosaic(raw_img, rows, cols);
        elseif bayertype == "rggb"
            rbg_img = linear_rggb_demosaic(raw_img, rows, cols);
        else
            error("Invalid bayertype");
        end

    elseif method == "nearest"
        if bayertype == "bggr"
            rbg_img = nearest_bggr_demosaic(raw_img, rows, cols);
        elseif bayertype == "gbrg"
            rbg_img = nearest_gbrg_demosaic(raw_img, rows, cols);
        elseif bayertype == "grbg"
            rbg_img = nearest_grbg_demosaic(raw_img, rows, cols);
        elseif bayertype == "rggb"
            rbg_img = nearest_rggb_demosaic(raw_img, rows, cols);
        else
            error("Invalid bayertype");
        end
    end
end

%% Functions for the linear method
function output_img = linear_bggr_demosaic(input_img, rows, cols)
    output_img = zeros(rows,cols,3);
    for i = 3:rows-2
        for j = 3:cols-2
            if (mod(i,2) == 0 && mod(j, 2) == 0)
                % Red pixel                   
                red_val = input_img(i,j);
                green_val = 0.25 * (input_img(i-1,j) + input_img(i,j-1) + input_img(i,j+1) + input_img(i+1,j));
                blue_val = 0.25 * (input_img(i-1,j-1) + input_img(i-1,j+1) + input_img(i+1,j-1) + input_img(i+1,j+1));
            elseif (mod(i,2) == 0 || mod(j, 2) == 0 && mod(i, 2) == 1)
                % Green pixel between blues
                red_val = 0.5 * (input_img(i-1,j) + input_img(i+1,j));
                green_val = input_img(i,j);
                blue_val = 0.5 * (input_img(i,j-1) + input_img(i,j+1));
            elseif (mod(i,2) == 0 || mod(j, 2) == 0 && mod(i, 2) == 0)
                % Green pixel between reds
                red_val = 0.5 * (input_img(i,j-1) + input_img(i,j+1));
                green_val = input_img(i,j);
                blue_val = 0.5 * (input_img(i-1,j) + input_img(i+1,j));
            else
                % Blue pixel
                red_val = 0.25 * (input_img(i-1,j-1) + input_img(i+1,j+1) + input_img(i-1,j+1) + input_img(i+1,j-1));
                green_val = 0.25 * (input_img(i-1,j) + input_img(i+1,j) + input_img(i,j-1) + input_img(i,j+1));
                blue_val = input_img(i,j);
            end
            % Store the color values in the output image
            output_img(i,j,1) = red_val;
            output_img(i,j,2) = green_val;
            output_img(i,j,3) = blue_val;
        end
    end
end

function output_img = linear_gbrg_demosaic(input_img, rows, cols)
    output_img = zeros(rows,cols,3);
    for i = 3:rows-2
        for j = 3:cols-2
            if (mod(i,2) == 1 && mod(j, 2) == 0)
                % Blue pixel                   
                red_val = 0.25 * (input_img(i-1,j-1) + input_img(i+1,j+1) + input_img(i-1,j+1) + input_img(i+1,j-1));
                green_val = 0.25 * (input_img(i-1,j) + input_img(i+1,j) + input_img(i,j-1) + input_img(i,j+1));
                blue_val = input_img(i,j);
            elseif (mod(i,2) == 0 && mod(j, 2) == 1)
                % Red pixel
                red_val = input_img(i,j);
                green_val = 0.25 * (input_img(i-1,j) + input_img(i+1,j) + input_img(i,j-1) + input_img(i,j+1));
                blue_val = 0.25 * (input_img(i-1,j-1) + input_img(i+1,j+1) + input_img(i-1,j+1) + input_img(i+1,j-1));
            else
                if (mod(i,2) == 0 && mod(j, 2) == 0)
                    % Green pixel between reds
                    red_val = 0.5 * (input_img(i,j-1) + input_img(i,j+1));
                    green_val = input_img(i,j);
                    blue_val = 0.5 * (input_img(i-1,j) + input_img(i+1,j));
                else
                    % Green pixel between blues
                    red_val = 0.5 * (input_img(i-1,j) + input_img(i+1,j));
                    green_val = input_img(i,j);
                    blue_val = 0.5 * (input_img(i,j-1) + input_img(i,j+1));
                end
            end
            % Store the color values in the output image
            output_img(i,j,1) = red_val;
            output_img(i,j,2) = green_val;
            output_img(i,j,3) = blue_val;
        end
    end
end

function output_img = linear_grbg_demosaic(input_img, rows, cols)
    output_img = zeros(rows,cols,3);
    for i = 3:rows-2
        for j = 3:cols-2
            if (mod(i,2) == 0 && mod(j, 2) == 1)
                % Blue pixel                   
                red_val = 0.25 * (input_img(i-1,j-1) + input_img(i+1,j+1) + input_img(i-1,j+1) + input_img(i+1,j-1));
                green_val = 0.25 * (input_img(i-1,j) + input_img(i+1,j) + input_img(i,j-1) + input_img(i,j+1));
                blue_val = input_img(i,j);
            elseif (mod(i,2) == 1 && mod(j, 2) == 0)
                % Red pixel
                red_val = input_img(i,j);
                green_val = 0.25 * (input_img(i-1,j) + input_img(i+1,j) + input_img(i,j-1) + input_img(i,j+1));
                blue_val = 0.25 * (input_img(i-1,j-1) + input_img(i+1,j+1) + input_img(i-1,j+1) + input_img(i+1,j-1));
            else
                if (mod(i,2) == 0 && mod(j, 2) == 0)
                    % Green pixel between blues
                    red_val = 0.5 * (input_img(i-1,j) + input_img(i+1,j));
                    green_val = input_img(i,j);
                    blue_val = 0.5 * (input_img(i,j-1) + input_img(i,j+1));
                else
                    % Green pixel between reds
                    red_val = 0.5 * (input_img(i,j-1) + input_img(i,j+1));
                    green_val = input_img(i,j);
                    blue_val = 0.5 * (input_img(i-1,j) + input_img(i+1,j));
                end
            end
            % Store the color values in the output image
            output_img(i,j,1) = red_val;
            output_img(i,j,2) = green_val;
            output_img(i,j,3) = blue_val;
        end
    end
end

function output_img = linear_rggb_demosaic(input_img, rows, cols)
    output_img = zeros(rows,cols,3);
    for i = 3:rows-2
        for j = 3:cols-2
            if (mod(i,2) == 0 && mod(j, 2) == 0)
                % Blue pixel                   
                red_val = 0.25 * (input_img(i-1,j-1) + input_img(i+1,j+1) + input_img(i-1,j+1) + input_img(i+1,j-1));
                green_val = 0.25 * (input_img(i-1,j) + input_img(i+1,j) + input_img(i,j-1) + input_img(i,j+1));
                blue_val = input_img(i,j);
            elseif (mod(j, 2) == 0 && mod(i, 2) == 1)
                % Green pixel between reds
                red_val = 0.5 * (input_img(i,j-1) + input_img(i,j+1));
                green_val = input_img(i,j);
                blue_val = 0.5 * (input_img(i-1,j) + input_img(i+1,j));
            elseif (mod(j, 2) == 1 && mod(i, 2) == 0)
                % Green pixel between blues
                red_val = 0.5 * (input_img(i-1,j) + input_img(i+1,j));
                green_val = input_img(i,j);
                blue_val = 0.5 * (input_img(i,j-1) + input_img(i,j+1));
            else
                % Red pixel
                red_val = input_img(i,j);
                green_val = 0.25 * (input_img(i-1,j) + input_img(i,j-1) + input_img(i,j+1) + input_img(i+1,j));
                blue_val = 0.25 * (input_img(i-1,j-1) + input_img(i-1,j+1) + input_img(i+1,j-1) + input_img(i+1,j+1));
            end
            % Store the color values in the output image
            output_img(i,j,1) = red_val;
            output_img(i,j,2) = green_val;
            output_img(i,j,3) = blue_val;
        end
    end
end

%% Functions for the nearest method
function output_img = nearest_bggr_demosaic(input_img, rows, cols)
    output_img = zeros(rows,cols,3);
    for i = 3:rows-2
        for j = 3:cols-2
            if (mod(i,2) == 0 && mod(j, 2) == 0)
                % Red pixel                   
                red_val = input_img(i,j);
                green_val = input_img(i-1,j);
                blue_val = input_img(i-1,j-1);
            elseif (mod(i,2) == 0 || mod(j, 2) == 0 && mod(i, 2) == 1)
                % Green pixel between blues
                red_val = input_img(i-1,j);
                green_val = input_img(i,j);
                blue_val = input_img(i,j-1);
            elseif (mod(i,2) == 0 || mod(j, 2) == 0 && mod(i, 2) == 0)
                % Green pixel between reds
                red_val = input_img(i,j-1);
                green_val = input_img(i,j);
                blue_val = input_img(i-1,j);
            else
                % Blue pixel
                red_val = input_img(i-1,j-1);
                green_val = input_img(i-1,j);
                blue_val = input_img(i,j);
            end
            % Store the color values in the output image
            output_img(i,j,1) = red_val;
            output_img(i,j,2) = green_val;
            output_img(i,j,3) = blue_val;
        end
    end
end

function output_img = nearest_gbrg_demosaic(input_img, rows, cols)
    output_img = zeros(rows,cols,3);
    for i = 3:rows-2
        for j = 3:cols-2
            if (mod(i,2) == 1 && mod(j, 2) == 0)
                % Blue pixel                   
                red_val = input_img(i-1,j-1);
                green_val = input_img(i-1,j);
                blue_val = input_img(i,j);
            elseif (mod(i,2) == 0 && mod(j, 2) == 1)
                % Red pixel
                red_val = input_img(i,j);
                green_val = input_img(i-1,j);
                blue_val = input_img(i-1,j-1);
            else
                if (mod(i,2) == 0 && mod(j, 2) == 0)
                    % Green pixel between reds
                    red_val = input_img(i,j-1);
                    green_val = input_img(i,j);
                    blue_val = input_img(i-1,j);
                else
                    % Green pixel between blues
                    red_val = input_img(i-1,j);
                    green_val = input_img(i,j);
                    blue_val = input_img(i,j-1);
                end
            end
            % Store the color values in the output image
            output_img(i,j,1) = red_val;
            output_img(i,j,2) = green_val;
            output_img(i,j,3) = blue_val;
        end
    end
end

function output_img = nearest_grbg_demosaic(input_img, rows, cols)
    output_img = zeros(rows,cols,3);
    for i = 3:rows-2
        for j = 3:cols-2
            if (mod(i,2) == 0 && mod(j, 2) == 1)
                % Blue pixel                   
                red_val = input_img(i-1,j-1);
                green_val = input_img(i-1,j);
                blue_val = input_img(i,j);
            elseif (mod(i,2) == 1 && mod(j, 2) == 0)
                % Red pixel
                red_val = input_img(i,j);
                green_val = input_img(i-1,j);
                blue_val = input_img(i-1,j-1);
            else
                if (mod(i,2) == 0 && mod(j, 2) == 0)
                    % Green pixel between blues
                    red_val = input_img(i-1,j);
                    green_val = input_img(i,j);
                    blue_val = input_img(i,j-1);
                else
                    % Green pixel between reds
                    red_val = input_img(i,j-1);
                    green_val = input_img(i,j);
                    blue_val = input_img(i-1,j);
                end
            end
            % Store the color values in the output image
            output_img(i,j,1) = red_val;
            output_img(i,j,2) = green_val;
            output_img(i,j,3) = blue_val;
        end
    end
end

function output_img = nearest_rggb_demosaic(input_img, rows, cols)
    output_img = zeros(rows,cols,3);
    for i = 3:rows-2
        for j = 3:cols-2
            if (mod(i,2) == 0 && mod(j, 2) == 0)
                % Blue pixel                   
                red_val = input_img(i-1,j-1);
                green_val = input_img(i-1,j);
                blue_val = input_img(i,j);
            elseif (mod(i,2) == 0 || mod(j, 2) == 0 && mod(i, 2) == 1)
                % Green pixel between reds
                red_val = input_img(i,j-1);
                green_val = input_img(i,j);
                blue_val = input_img(i-1,j);
            elseif (mod(i,2) == 0 || mod(j, 2) == 0 && mod(i, 2) == 0)
                % Green pixel between blues
                red_val = input_img(i-1,j);
                green_val = input_img(i,j);
                blue_val = input_img(i,j-1);
            else
                % Red pixel
                red_val = input_img(i,j);
                green_val = input_img(i-1,j);
                blue_val = input_img(i-1,j-1);
            end
            % Store the color values in the output image
            output_img(i,j,1) = red_val;
            output_img(i,j,2) = green_val;
            output_img(i,j,3) = blue_val;
        end
    end
end