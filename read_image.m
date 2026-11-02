function I = read_image(filename)
    [I, map] = imread(filename);
    if ~isempty(map)
        I = ind2rgb(I, map);
    end
    if isa(I, 'uint8')
        I = double(I) / 255;
    elseif isa(I, 'uint16')
        I = double(I) / 65535;
    else
        I = double(I);
    end
    if size(I, 3) == 4
        I = I(:, :, 1:3);
    end
end
