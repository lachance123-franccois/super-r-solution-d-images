function [lr, hr] = degrade(hr, scale)
%   L'image HR est recadrée à un multiple de scale, puis réduite par
%   interpolation bicubique (avec anti-repliement).
    H = floor(size(hr, 1) / scale) * scale; 
    W = floor(size(hr, 2) / scale) * scale;
    hr = hr(1:H, 1:W, :); % on jette ici les valeur restante en bas et a droit pour avoir notre image basse resolution
    lr = imresize(hr, [H W]/scale ,'bicubic');
end
