function mask = wbmask(m, n, wbcoeffs, bayertype)
% Makes a white-balance multiplicative mask with RGB white balance multipliers 
% wbcoeffs = [R_scale G_scale B_scale].

% initialize to all green values
mask = wbcoeffs(2)*ones(m,n); 

% bayertype value indicates the Bayer arrangement: 'rggb','gbrg','grbg','bggr'
switch bayertype
    case 'rggb'
        mask(1:2:end,1:2:end) = wbcoeffs(1);   
        mask(2:2:end,2:2:end) = wbcoeffs(3);    
    case 'bggr'
        mask(2:2:end,2:2:end) = wbcoeffs(1);    
        mask(1:2:end,1:2:end) = wbcoeffs(3);    
    case 'grbg'
        mask(1:2:end,2:2:end) = wbcoeffs(1);    
        mask(1:2:end,2:2:end) = wbcoeffs(3);    
    case 'gbrg'
        mask(2:2:end,1:2:end) = wbcoeffs(1);    
        mask(1:2:end,2:2:end) = wbcoeffs(3);    
end
end
