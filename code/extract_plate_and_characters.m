function [global_mask, img_plate_color, img_characters, x_ranges, y_range] = ...
    extract_plate_and_characters(img)
addpath('./utilities');
addpath('./utilities/detection');
addpath('./utilities/orientation');
addpath('./utilities/segmentation');
addpath('./utilities/transformation');

img_denoised = med_filt_rgb(img);
img_gray = grayscale(img_denoised);
img_eq = adapthisteq(img_gray);
valid_mask = reshape([1:numel(img_eq)], size(img_eq));

% 使用 canny 检测粗边缘，从中估计水平线的角度
threshold = [0.2 0.6];
img_edge = canny_no_border(img_eq, threshold);
[H, T, ~] = hough(img_edge);
theta_index = get_hline_theta_index(H);

% 将水平线旋转至水平
rot_angle = -sign(T(theta_index)) * (90 - abs(T(theta_index)));
img_eq_rot = rotate_image(img_eq, rot_angle);
img_color_rot = rotate_image(img_denoised, rot_angle);
valid_mask = rotate_image(valid_mask, rot_angle, 'nearest');

% 使用 canny 检测细边缘
threshold_detail = [0.1, 0.3];
img_edge_detail = canny_no_border(img_eq_rot, threshold_detail);

% 执行 Hough 变换，仅提取水平或垂直线
[H, T, R] = hough(img_edge_detail);
H_horizontal = get_H_valid(H, 1, 3);
H_vertical = get_H_valid(H, 91, 3);
v_lines = get_vertical_lines(img_edge_detail, H_vertical, T, R);
h_lines = get_horizontal_lines(img_edge_detail, H_horizontal, T, R);

% 利用提取到的水平、垂直线估计汽车所在的大致范围
rect_car = get_car_rect(h_lines, v_lines);
img_car = rotate_image(img_gray, rot_angle);
img_car = img_car(rect_car.point1(2):rect_car.point2(2), ...
    rect_car.point1(1):rect_car.point2(1), :);
img_car_eq = adapthisteq(img_car, 'NumTiles', [16 16], 'ClipLimit', 0.01);
img_car_color = img_color_rot(rect_car.point1(2):rect_car.point2(2), ...
    rect_car.point1(1):rect_car.point2(1), :);
valid_mask = valid_mask(rect_car.point1(2):rect_car.point2(2), ...
    rect_car.point1(1):rect_car.point2(1));

% 粗提取车牌 mask
car_edge_detail = edge(img_car_eq, 'Canny', [0.2 0.45]);
plate_mask = get_plate_mask(car_edge_detail);

% 拟合矩形，扩大选区，精准提取车牌区域
plate_boxes_rough = get_bounding_rects(plate_mask);
plate_boxes_ext = get_bounding_rects(plate_mask, [1.35, 1.15]);
for k = 1:length(plate_boxes_ext)
    % 粗提获得的彩色车牌图像
    img_plate_color_rough = ...
        img_car_color(plate_boxes_rough{k}.point1(2):plate_boxes_rough{k}.point2(2), ...
        plate_boxes_rough{k}.point1(1):plate_boxes_rough{k}.point2(1), :);

    % 扩大选区后的彩色车牌图像
    img_plate_color_ext = ...
        img_car_color(plate_boxes_ext{k}.point1(2):plate_boxes_ext{k}.point2(2), ...
        plate_boxes_ext{k}.point1(1):plate_boxes_ext{k}.point2(1), :);
    global_mask_cur = ...
        valid_mask(plate_boxes_ext{k}.point1(2):plate_boxes_ext{k}.point2(2), ...
        plate_boxes_ext{k}.point1(1):plate_boxes_ext{k}.point2(1));
    
    % 利用色彩与边缘信息提取精确的彩色车牌图像
    plate_rect = get_accurate_plate_rect(img_plate_color_rough, img_plate_color_ext);
    img_plate_color = ...
        img_plate_color_ext(plate_rect.point1(2):plate_rect.point2(2), ...
        plate_rect.point1(1):plate_rect.point2(1), :);
    valid_mask_cur = ...
        global_mask_cur(plate_rect.point1(2):plate_rect.point2(2), ...
        plate_rect.point1(1):plate_rect.point2(1));
         
    [img_characters, x_ranges, y_range] = get_character_pos(img_plate_color);
    if ~isempty(x_ranges)
        global_mask = generate_global_mask(size(img_eq), valid_mask_cur);
        return;
    end
end
end
