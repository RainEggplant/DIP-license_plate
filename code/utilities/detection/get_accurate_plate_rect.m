function rect = get_accurate_plate_rect(img_plate_color_rough, img_plate_color_ext)
height = size(img_plate_color_ext, 1);
width = size(img_plate_color_ext, 2);
[h, s, v] = rgb2hsv(img_plate_color_rough);
    
if sum(s <= 0.05, 'all') > 0.35 * numel(s)
    % ��Ϊ���Ʊ���Ϊ��ɫ���߳����˹���
    % �Ա��Ͷ�Ϊ������������ɸѡ
    vv = v(s <= 0.05);
    v_range = [mean(vv) - 2 * std(vv), mean(vv) + 2 * std(vv)];
    img_filt = in_range(rgb2hsv(img_plate_color_ext), [0, 0, v_range(1)], [1, 0.05, v_range(2)]);
else
    % ��ɫ�����������ɫ��Ϊ������������ɸѡ
    hh = h(s > 0.05);
    f = figure('visible','off');
    hist_h = histogram(hh(:), [0:1/24:1]);
    [~, index] = max(hist_h.BinCounts);
    close(f);
    h_range = [(index - 1) / 24, index / 24];
    vv = v(s > 0.05 & h > h_range(1) & h < h_range(2));
    v_range = [mean(vv) - 2 * std(vv), mean(vv) + 2 * std(vv)];
    img_filt = in_range(rgb2hsv(img_plate_color_ext), [h_range(1), 0.05, v_range(1)], [h_range(2), 1, v_range(2)]);
end

% ����ȡ���ƵĴ�������
img_op = imopen(img_filt, [1 1]);

img_cl = imclose(img_op, ones(round(height / 4), round(width / 8)));
img_op = imopen(img_cl, ones(round(height / 4), round(width / 8)));

img_cl = imclose(img_op, ones(round(height / 3), round(width / 6)));
img_op = imopen(img_cl, ones(round(height / 3), round(width / 6)));

% �������ϻ�õ�ѡ��
boxes = get_bounding_rects(img_op, [1.15, 1.02]);
box_id = get_max_area_rect_id(boxes, 2.5);

if box_id == 0
    rect.point1 = [-1, -1];
    return;
end

% ����ѡ���ڵı�Ե��Ϣ
im_gray = adapthisteq(grayscale(img_plate_color_ext));
im_edge = edge(im_gray, 'Canny', [0.4 0.7]);
im_valid_edge = zeros(height, width);
im_valid_edge(boxes{box_id}.point1(2):boxes{box_id}.point2(2), ...
    boxes{box_id}.point1(1):boxes{box_id}.point2(1)) = 1;
im_valid_edge = im_valid_edge & im_edge;

% ����ȡ��������
img_op = img_filt | im_valid_edge;
img_op = imopen(img_op, [1 1]);

img_cl = imclose(img_op, ones(round(height / 4), round(width / 8)));
img_op = imopen(img_cl, ones(round(height / 4), round(width / 8)));

img_cl = imclose(img_op, ones(round(height / 3), round(width / 6)));
img_op = imopen(img_cl, ones(round(height / 3), round(width / 6)));

acc_plate_boxes = get_bounding_rects(img_op, [1.01, 1.02]);
box_id = get_max_area_rect_id(acc_plate_boxes, 2.5);

if box_id == 0
    rect.point1 = [-1, -1];
    return;
end

rect = acc_plate_boxes{box_id};
end
