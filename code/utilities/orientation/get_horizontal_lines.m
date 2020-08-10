function lines = get_horizontal_lines(img_edge, H_horizontal, T, R)
scale = size(H_horizontal, 1);
P_horizontal = houghpeaks(H_horizontal, 30, 'threshold', ceil(0.15 * max(H_horizontal(:))));
lines = houghlines(img_edge, T, R, P_horizontal, 'FillGap', ...
    scale / 190, 'MinLength', scale / 40);
end
