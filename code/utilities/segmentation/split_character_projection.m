function seg_pos = split_character_projection(projection)
len = length(projection);
character_len = 0.055 * len;
threshold = 0;

while 1
    sections = get_sections(projection, threshold);
    if length(sections) >= 6
        seg_pos = [];

        if sections{1}(1) > character_len
            seg_pos(end + 1) = 1;
        end

        for sec_id = 2:length(sections)
            if min_middle(sections{sec_id}) - min_middle(sections{sec_id - 1}) > character_len
                seg_pos(end + 1) = min_middle(sections{sec_id - 1});
            end
        end

        seg_pos(end + 1) = min_middle(sections{sec_id});

        if len - sections{end}(2) + 1 > character_len
            seg_pos(end + 1) = len;
        end

        if length(seg_pos) >= 8
            break;
        end   
    end
    
    threshold = threshold + 0.005;
    if threshold > 0.2
        break;
    end
end

seg_pos = round(seg_pos);

    function pos_mid = min_middle(section)
        [val, pos] = min(projection(section(1):section(2)));
        pos = section(1) + pos - 1;
        pos_right = pos + 1;
        while pos_right < section(2) && projection(pos_right) == val
            pos_right = pos_right + 1;
        end
        pos_right = pos_right - 1;
        pos_mid = round((pos + pos_right) / 2);
    end
end

function sections = get_sections(data, threshold)
sections = {};
len = length(data);

start_pos = 0;
for x = 1:len
    if start_pos == 0 && data(x) <= threshold
        start_pos = x;
    elseif start_pos ~= 0 && data(x) > threshold
        sections{end + 1} = [start_pos, x - 1];
        start_pos = 0;
    end
end

if start_pos ~= 0
    sections{end + 1} = [start_pos, x];
end
end
