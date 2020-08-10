function [img_characters, x_ranges, y_range] = split_character(BW)
area_threshold = round(numel(BW) * 0.004);
img = bwareaopen(BW, area_threshold);
img = medfilt2(img, [3, 3]);
img_characters = img;
img = imopen(img, ones(2, 2));

if ~check_validity(img)
    x_ranges = {};
    y_range = [];
    return;
end

y_range = get_y_range(img);
img_cut = img(y_range(1):y_range(2), :);

projection = calc_projection(img_cut);
seg_pos = split_character_projection(projection);

ch_width = seg_pos(2:end) - seg_pos(1:end - 1);
ch_width_median = median(ch_width);

x_ranges = {};
for n = 1:length(ch_width)
    if ch_width(n) > 1.6 * ch_width_median
        x_seg = split_character_cc(img_cut(:, seg_pos(n):seg_pos(n + 1)), 0.5 * ch_width_median);
        if isempty(x_seg)
            x_ranges{end + 1} = [seg_pos(n), seg_pos(n + 1)];
        else
            for k = 1:length(x_seg)
                x_ranges{end + 1} = [x_seg{k}(1), x_seg{k}(2)] + seg_pos(n) - 1;
            end
        end
    else
        x_ranges{end + 1} = [seg_pos(n), seg_pos(n + 1)];
    end
end
end
