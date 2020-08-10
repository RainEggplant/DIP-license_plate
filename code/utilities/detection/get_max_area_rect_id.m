function rect_id = get_max_area_rect_id(rects, wh_ratio_threshold)
area = 0;
rect_id = 0;

for k = 1:length(rects)
    rect_width = rects{k}.point2(1) - rects{k}.point1(1);
    rect_height = rects{k}.point2(2) - rects{k}.point1(2);
    if nargin > 1 && rect_width / rect_height < wh_ratio_threshold
        % 长宽比不符合标准，跳过
        continue;
    end
    
    area_cur = rect_width * rect_height;
    if area_cur > area
        rect_id = k;
        area = area_cur;
    end
end
end
