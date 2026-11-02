function p = psnr_db(ref, est, peak)
%PSNR_DB Peak Signal-to-Noise Ratio en dB (
%   p = psnr_db(ref, est)          crête = 1 (images dans [0, 1])
%   p = psnr_db(ref, est, peak)    ex. peak = 255
    if nargin < 3
        peak = 1;
    end
    mse = mean((double(ref(:)) - double(est(:))).^2);
    p = 10 * log10(peak^2 / mse);
end
