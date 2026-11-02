function s = ssim_index(ref, est, peak)
%SSIM_INDEX SSIM moyen (Wang et al., 2004), sans toolbox.
%   Fenêtre gaussienne 11x11 (sigma = 1.5), K1 = 0.01, K2 = 0.03.
%   Images en niveaux de gris (pour du RGB, moyenner sur les canaux).
%   peak = 1 par défaut (images dans [0, 1]).
    if nargin < 3
        peak = 1;
    end
    x = double(ref);
    y = double(est);
    C1 = (0.01 * peak)^2;
    C2 = (0.03 * peak)^2;

    g = exp(-(-5:5).^2 / (2 * 1.5^2));
    g = g / sum(g);
    filt = @(z) conv2(g, g, padarray_reflect(z, 5), 'valid');

    mx = filt(x);
    my = filt(y);
    sxx = filt(x .* x) - mx.^2;
    syy = filt(y .* y) - my.^2;
    sxy = filt(x .* y) - mx .* my;
    map = ((2 * mx .* my + C1) .* (2 * sxy + C2)) ./ ((mx.^2 + my.^2 + C1) .* (sxx + syy + C2));
    map = map(6:end-5, 6:end-5);
    s = mean(map(:));
end
