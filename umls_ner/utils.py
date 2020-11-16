import re

def get_pattern(text, words, specify_pattern, start_end_blank_pattern, item_pattern):
    pattern_list = list(r.group() for r in specify_pattern.finditer(text))
    if pattern_list:
        len_words = len(words)
        matched_idx = set()
        for i, unit in enumerate(pattern_list):
            start = None
            result = start_end_blank_pattern.sub('', unit)
            result = item_pattern.sub('', result)
            len_result = len(result.split(' '))
            for idx in range(len_words-len_result+1):
                if result in ' '.join(words[idx:idx+len_result]) and idx not in matched_idx:
                    pattern_parentheses = re.search(rf'\(.*?{result}.*?\)', text)
                    if pattern_parentheses:
                        start = len(text.split(pattern_parentheses.group())[0].split(' '))-1
                    else:
                        start = idx
                    matched_idx.add(start)    
                    break
            if start is not None:
                pattern_list[i] = (start, result)
            else:
                raise IndexError('Do not match start index of specify pattern.')
    return pattern_list

def remove_parentheses(text, parentheses_pattern, blank_pattern):
    result = parentheses_pattern.sub('', text) # 排除括弧裡的資訊
    result = blank_pattern.sub(' ', result)
    return result

def generate_cls_dict(cls_dict, collect, words, ner):
    for item in collect:
        ent = ' '.join(words[item[0]:item[1]])
        if ent not in cls_dict:
            cls_dict[ent] = ner
    return cls_dict

def combine_Collect(words, collect, preposition, att2sym_pattern, target):
    collect_all = sorted(collect, key=lambda x: x[0])
    size = len(collect_all)
    result = []
    temp = []
    for idx, (start, end, attri) in enumerate(collect_all):
        temp.append(collect_all[idx])
        if idx != size-1 and all(words[i] in preposition for i in range(end, collect_all[idx+1][0])):
            continue
        if target == 'Location' and (len(temp) > 1 or attri == target or (end - start) > 1):
            result.append((temp[0][0], temp[-1][1], target))
        if target == 'Symptom':
            if any(ele[2] == 'Symptom' for ele in temp):
                result.append((temp[0][0], temp[-1][1], target))
            else:
                for pattern in att2sym_pattern:
                    if pattern.search(' '.join(words[temp[0][0]:temp[-1][1]])):
                        result.append((temp[0][0], temp[-1][1], target))
                        break
        temp = []
    return result

def pattern_item_match(pattern_list, collect_item_combine): # pattern_list -> (index, string)
    if pattern_list:
        for i, patt in enumerate(pattern_list):
            item_match = ''
            if collect_item_combine:
                item_dist = [(item, min(abs(item[0]-patt[0]), abs(item[1]-patt[0]))) for item in collect_item_combine]
                item_match = min(item_dist, key=lambda x: x[1])[0]
            pattern_list[i] = (patt[1], item_match) # (string, (start, end, attri))

def collect_colon_item(words, label_seq, collect_loc_combine, collect_sym_combine):
    collect_colon_loc_sym = []
    if ':' in words:
        colon_idx = words.index(':')
        colon_item = all(label != 'Other' or words[i] == 'and' for i, label in enumerate(label_seq[:colon_idx]))
        if colon_item:
            for item in sorted(collect_loc_combine + collect_sym_combine, key=lambda x:x[0]):
                if item[1] <= colon_idx:
                    collect_colon_loc_sym.append(item)
            collect_loc_combine = [item for item in collect_loc_combine if item not in collect_colon_loc_sym]
            collect_sym_combine = [item for item in collect_sym_combine if item not in collect_colon_loc_sym]
    return collect_colon_loc_sym, collect_loc_combine, collect_sym_combine

def filter_output(collect, words):
    letter_pattern = re.compile(r'[a-zA-Z]')
    result = [item for item in collect if letter_pattern.search(' '.join(words[item[0]:item[1]]))]
    return result

def get_residual_num_mod(collect_target, collect_sym_combine):
    residual_list = []
    for item in collect_target:
        residual = True
        for sym in collect_sym_combine:
            if sym[0] <= item[0] and item[1] <= sym[1]:
                residual = False
                break
        if residual:
            residual_list.append(item)
    return residual_list

def get_comma(words, collect_loc_combine):
    result = []
    for idx, word in enumerate(words):
        if word == ',' or word == ';':
            block = True
            for item in collect_loc_combine:
                if item[0] <= idx and idx < item[1]:
                    block = False
                    break
            if block:
                result.append((idx, idx+1, 'comma'))
    return result

def split_by_comma(collect_all):
    sort_collect = sorted(collect_all, key=lambda x: x[0])
    size = len(sort_collect) 
    idx_list = [idx + 1 for idx, val in enumerate(sort_collect) if val[2] == 'comma']
    if idx_list:
        res = [sort_collect[i: j] for i, j in zip([0] + idx_list, idx_list + ([size] if idx_list[-1] != size else []))]
        return res
    else:
        return [sort_collect]

def get_adjacent_pair(part):
    adjacent_pair = []
    residual_part = []
    part = [ele for ele in part if ele[2] != 'comma']
    size = len(part)
    temp = False
    for idx, (start, end, attr) in enumerate(part):
        if temp:
            temp = False
            continue
        if idx != size-1 and end == part[idx+1][0]:
            pair = (part[idx], part[idx+1])
            if attr == 'Symptom':
                pair = (part[idx+1], part[idx])
            adjacent_pair.append(pair)
            temp = True
        else:
            residual_part.append(part[idx])
    return adjacent_pair, residual_part

def get_residual_pair(buff, part, words, preposition, pair_set, mode):
    new_buff = []
    for ele in buff:
        block = True
        tar_idx = part.index(ele)
        if mode == 'loc':
            pair_attr = 'Symptom'
        elif mode == 'sym':
            pair_attr = 'Location'
        else:
            raise ValueError('Not Found mode')
        idx_list = [idx for idx, item in enumerate(part) if item[2] == pair_attr]
        sort_idx_list = sorted([(x, abs(x-tar_idx)) for x in idx_list], key=lambda x:x[1])
        for near_idx, _ in sort_idx_list:
            mid_words = set(words[i] for i in range(part[min(tar_idx, near_idx)][1], part[max(tar_idx, near_idx)][0]))
            if preposition & mid_words:
                pair = (part[near_idx], part[tar_idx])
                if mode == 'loc':
                    pair = (part[tar_idx], part[near_idx])
                pair_set.add(pair)
                block = False
                break
        if block:
            new_buff.append(ele)
    return new_buff, pair_set

def retrieve_att2sym(sym, sym_ents, att_ents, att2sym_pattern, start_end_blank_pattern):
    if not sym_ents:
        for pattern in att2sym_pattern:
            temp_sym = re.search(pattern, sym)
            if temp_sym:
                sym_ents.append(temp_sym.group())
                for i, att_ent in enumerate(list(att_ents)):
                    if att_ent in temp_sym.group():
                        att_ents.remove(att_ent)

def collect_ner_ent(ner_set, ents, ner):
    pattern = re.compile(r'\d')
    for ent in ents:
        if not pattern.search(ent):
            ner_set.add((ent, ner))
    return ner_set

def detect_symptom(sym, sent, normal_sym_vocab, negative_key_pattern_forw, negative_key_pattern_backw,
                    uncertain_key_pattern_forw, uncertain_key_pattern_backw, Negsd):
    if sym.lower() in normal_sym_vocab:
        return 'No'
    sent_rep = sent.lower().replace(sym, 'C1234567')
    sym = re.sub(r'([()])', r'\\\1', sym)
    sent_split = re.split(rf'{sym}(\s|$)', sent)
    sent_split_forward = re.split(',', sent_split[0])[-1]
    sent_split_backward = re.split(',', sent_split[-1])[0]
    if any(pattern.search(sent_split_forward) for pattern in uncertain_key_pattern_forw) or any(pattern.search(sent_split_backward) for pattern in uncertain_key_pattern_backw):
        return 'Uncertain'
    if Negsd.detect(sent_rep) or any(pattern.search(sent_split_forward) for pattern in negative_key_pattern_forw) or any(pattern.search(sent_split_backward) for pattern in negative_key_pattern_backw):
        return 'No'
    return 'Yes'

def extract_att_from_sym(sym2att_pattern, sym, pair_info, start_end_blank_pattern):
    for pattern in sym2att_pattern:
        new_att = pattern.search(sym)
        if new_att:
            pair_info.att_ents.add(new_att.group())
            pair_info.sym_ents.remove(sym)
            sym = sym.replace(new_att.group(), '')
            sym = start_end_blank_pattern.sub('', sym)
            pair_info.sym_ents.append(sym)
    return sym

def get_span(tar_ents, tar_tuple, words):
    tar_span_list = []
    for tar in sorted(tar_ents, key=lambda x: len(x.split(' ')), reverse=True):
        if tar in ' '.join(words[tar_tuple[0]:tar_tuple[1]]):
            tar_words = tar.split(' ')
            for i in range(len(words[tar_tuple[0]:tar_tuple[1]])-len(tar_words)+1):
                if tar_words == words[tar_tuple[0]:tar_tuple[1]][i:i+len(tar_words)]:
                    tar_span = (i+tar_tuple[0], i+tar_tuple[0]+len(tar_words))
                    if all(tar_span[0] >= end or tar_span[0] < start for start, end in tar_span_list):
                        tar_span_list.append(tar_span)
    tar_span_list = [str(ele) for ele in tar_span_list]
    return tar_span_list
