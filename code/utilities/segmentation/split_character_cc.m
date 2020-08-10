function x_seg = split_character_cc(BW, threshold_width)
x_seg = {};
stats = regionprops(BW, 'BoundingBox');
for n = 1:length(stats)
    if stats(n).BoundingBox(3)< threshold_width
        continue;
    end
    x_seg{end + 1} = round([stats(n).BoundingBox(1), stats(n).BoundingBox(1) + stats(n).BoundingBox(3) - 1]);
end
end
