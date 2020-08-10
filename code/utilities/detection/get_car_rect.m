function rect = get_car_rect(h_lines, v_lines)
h_left = sort(arrayfun(@(x) min(x.point1(1), x.point2(1)), h_lines));
h_right = sort(arrayfun(@(x) max(x.point1(1), x.point2(1)), h_lines), 'descend');
h_top = sort(arrayfun(@(x) min(x.point1(2), x.point2(2)), h_lines));
h_bottom = sort(arrayfun(@(x) max(x.point1(2), x.point2(2)), h_lines), 'descend');
v_left = sort(arrayfun(@(x) min(x.point1(1), x.point2(1)), v_lines));
v_right = sort(arrayfun(@(x) max(x.point1(1), x.point2(1)), v_lines), 'descend');

if h_left(1) < v_left(1)
    if h_left(2) < v_left(1)  % --|
        left_most = h_left(2);
    else  % -|-
        left_most = min(h_left(2), v_left(2));
    end
else
    if h_left(1) < v_left(2)  % |-|
        left_most = min(h_left(2), v_left(2));
    else  % ||-
        left_most = min(h_left(1), v_left(3));
    end
end

if h_right(1) > v_right(1)
    if h_right(2) > v_right(1)
        right_most = h_right(2);
    else  % -|-
        right_most = max(h_right(2), v_right(2));
    end
else
    if h_right(1) > v_right(2)
        right_most = max(h_right(2), v_right(2));
    else  % ||-
        right_most = max(h_right(1), v_right(3));
    end
end

top_most = h_top(1);
bottom_most = h_bottom(1);

rect.point1 = [left_most, top_most];
rect.point2 = [right_most, bottom_most];
end
