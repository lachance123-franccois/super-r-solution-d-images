function Y = rgb_to_ycbcr(I)
%RGB_TO_YCBCR Conversion RGB -> YCbCr pleine échelle (JPEG, BT.601), I dans [0, 1].
    M = [ 0.299     0.587     0.114;
         -0.168736 -0.331264  0.5;
          0.5      -0.418688 -0.081312];
    [H, W, ~] = size(I);
    Y = reshape(reshape(I, [], 3) * M', H, W, 3);
    Y(:, :, 2:3) = Y(:, :, 2:3) + 0.5;
end
