function [FL, ii, jj] = feature_patches(mr, ps, step)
    F = sr_features(mr);
    FL = [];
    for k = 1:4
        [Pk, ii, jj] = extract_patches(F(:, :, k), ps, step);
        FL = [FL; Pk];
    end
end
