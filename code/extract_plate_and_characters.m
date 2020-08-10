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

% ʹ�� canny ���ֱ�Ե�����й���ˮƽ�ߵĽǶ�
threshold = [0.2 0.6];
img_edge = canny_no_border(img_eq, threshold);
[H, T, ~] = hough(img_edge);
theta_index = get_hline_theta_index(H);

% ��ˮƽ����ת��ˮƽ
rot_angle = -sign(T(theta_index)) * (90 - abs(T(theta_index)));
img_eq_rot = rotate_image(img_eq, rot_angle);
img_color_rot = rotate_image(img_denoised, rot_angle);
valid_mask = rotate_image(valid_mask, rot_angle, 'nearest');

% ʹ�� canny ���ϸ��Ե
threshold_detail = [0.1, 0.3];
img_edge_detail = canny_no_border(img_eq_rot, threshold_detail);

% ִ�� Hough �任������ȡˮƽ��ֱ��
[H, T, R] = hough(img_edge_detail);
H_horizontal = get_H_valid(H, 1, 3);
H_vertical = get_H_valid(H, 91, 3);
v_lines = get_vertical_lines(img_edge_detail, H_vertical, T, R);
h_lines = get_horizontal_lines(img_edge_detail, H_horizontal, T, R);

% ������ȡ����ˮƽ����ֱ�߹����������ڵĴ��·�Χ
rect_car = get_car_rect(h_lines, v_lines);
img_car = rotate_image(img_gray, rot_angle);
img_car = img_car(rect_car.point1(2):rect_car.point2(2), ...
    rect_car.point1(1):rect_car.point2(1), :);
img_car_eq = adapthisteq(img_car, 'NumTiles', [16 16], 'ClipLimit', 0.01);
img_car_color = img_color_rot(rect_car.point1(2):rect_car.point2(2), ...
    rect_car.point1(1):rect_car.point2(1), :);
valid_mask = valid_mask(rect_car.point1(2):rect_car.point2(2), ...
    rect_car.point1(1):rect_car.point2(1));

% ����ȡ���� mask
car_edge_detail = edge(img_car_eq, 'Canny', [0.2 0.45]);
plate_mask = get_plate_mask(car_edge_detail);

% ��Ͼ��Σ�����ѡ������׼��ȡ��������
plate_boxes_rough = get_bounding_rects(plate_mask);
plate_boxes_ext = get_bounding_rects(plate_mask, [1.35, 1.15]);
for k = 1:length(plate_boxes_ext)
    % �����õĲ�ɫ����ͼ��
    img_plate_color_rough = ...
        img_car_color(plate_boxes_rough{k}.point1(2):plate_boxes_rough{k}.point2(2), ...
        plate_boxes_rough{k}.point1(1):plate_boxes_rough{k}.point2(1), :);

    % ����ѡ����Ĳ�ɫ����ͼ��
    img_plate_color_ext = ...
        img_car_color(plate_boxes_ext{k}.point1(2):plate_boxes_ext{k}.point2(2), ...
        plate_boxes_ext{k}.point1(1):plate_boxes_ext{k}.point2(1), :);
    global_mask_cur = ...
        valid_mask(plate_boxes_ext{k}.point1(2):plate_boxes_ext{k}.point2(2), ...
        plate_boxes_ext{k}.point1(1):plate_boxes_ext{k}.point2(1));
    
    % ����ɫ�����Ե��Ϣ��ȡ��ȷ�Ĳ�ɫ����ͼ��
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
