import re


class Dataset(object):
    def __init__(self, ori_sent, sent, UMLS_TUI, UMLS_CUI, CUI_sets, CUI_cls,
                 stop_word, tui_dict, tui_dict_level, vocab_type, unit_list, 
                 composition_list, mod_muti_word_set, sym_muti_word_set, loc_muti_word_set):
        self.ori_sent = ori_sent
        self.sent = sent
        self.words = [word for word in sent.split(' ')]
        self.stop_word = stop_word
        self.unit_list = unit_list
        self.composition_list = composition_list
        self.label_seq = [vocab_type[word.lower()] for word in self.words]

        for start_idx, end_idx, tui_list in UMLS_TUI(self.words):
            label = tui_dict[max(tui_dict_level[tui] for tui in tui_list)]
            for j in range(start_idx, end_idx):
                if self.words[j].lower() in self.stop_word:
                    continue
                if self.label_seq[j] == 'Other' or label != 'Symptom':
                    if j == start_idx:
                        self.label_seq[j] = 'B-'+label
                    else:
                        self.label_seq[j] = 'I-'+label

        self.CUI_label(UMLS_CUI, CUI_sets, CUI_cls)

        self.muti_word_match(mod_muti_word_set, mode='Modify')
        self.muti_word_match(sym_muti_word_set, mode='Symptom')
        self.muti_word_match(loc_muti_word_set, mode='Location')

    def CUI_label(self, UMLS_CUI, CUI_sets, CUI_cls):
        len_CUI_sets = len(CUI_sets)
        for start_idx, end_idx, cui_list in UMLS_CUI(self.words):
            for i in range(start_idx, end_idx):
                if self.words[i].lower() in self.stop_word:
                    continue
                for j in range(len_CUI_sets):
                    if any(cui in CUI_sets[j] for cui in cui_list):
                        if i == start_idx:
                            self.label_seq[i] = 'B-'+CUI_cls[j]
                        else:
                            self.label_seq[i] = 'I-'+CUI_cls[j]
                        break

    def muti_word_match(self, muti_word_set, mode):
        sent_lower = self.sent.lower()
        words = [word.lower() for word in self.words]
        len_words = len(self.words)
        for muti_word in muti_word_set:
            if muti_word in sent_lower:
                muti_words = muti_word.split(' ')
                len_muti_words = len(muti_words)
                for idx in range(len_words-len_muti_words+1):
                    if words[idx:idx+len_muti_words] == muti_words:
                        for i in range(idx, idx+len_muti_words):
                            if i == idx:
                                self.label_seq[i] = 'B-'+mode
                            else:
                                self.label_seq[i] = 'I-'+mode
                            
    def convert_bio(self):
        if self.label_seq:
            temp = None
            for i, label in enumerate(self.label_seq):
                if label == 'Other':
                    temp = None
                    continue
                char = 'I-'
                if temp is None or label != temp:
                    temp = label
                    char = 'B-'
                self.label_seq[i] = char+label
    
    def collect_output(self, attri):
        result = []
        temp = []
        collect = tuple((idx , label) for idx , label in enumerate(self.label_seq) if attri in label)
        size = len(collect)
        for i, (idx, label) in enumerate(collect):
            temp.append((idx , label))
            if i == size-1 or 'B-' in collect[i+1][1]:
                result.append((temp[0][0], temp[-1][0]+1, attri))
                temp = []
        return result


class PairDataset(object):
    def __init__(self, words, match_list, cls_dict, blank_pattern, start_end_blank_pattern, preposition_loc,
                        pair_loc=None, pair_sym=None, residual_num=None, residual_mod=None):
        self.loc = ' '.join(words[pair_loc[0]:pair_loc[1]]) if pair_loc else ''
        self.resi_num = self.num_mod2sym(residual_num, pair_sym, words) if pair_sym and residual_num else []
        self.resi_mod = self.num_mod2sym(residual_mod, pair_sym, words) if pair_sym and residual_mod else []
        self.sym = ' '.join(self.resi_num + self.resi_mod + words[pair_sym[0]:pair_sym[1]]) if pair_sym else ''
        self.match_pattern = []
        if match_list:
            for ele in match_list:
                if ele[1] == pair_loc or ele[1] == pair_sym:
                    self.match_pattern.append(ele[0])
        self.sym_ents = []
        self.loc_ents = set()
        self.att_ents = set()
        
        for key, value in cls_dict.items():
            if key == 'change' or key == 'changes':
                continue
            if value == 'Att' and key in self.sym or key in self.match_pattern:
                self.att_ents.add(key)
                self.sym = self.sym.replace(key, '', 1)
                self.sym = blank_pattern.sub(' ', self.sym)
                self.sym = start_end_blank_pattern.sub('', self.sym)
            if value == 'Att' and key in self.loc:
                self.att_ents.add(key)
                self.loc = self.loc.replace(key, '')
                self.loc = blank_pattern.sub(' ', self.loc)
                self.loc = start_end_blank_pattern.sub('', self.loc)
        
        if self.sym in {'of', 'with', '/', 'in'}:
            self.syms = []
        else:
            self.sym = re.sub(r'\sof\s|^of\s|\sof$', ' ', self.sym)
            self.sym = re.sub(r'^\s*in\s|\sin\s*$', '', self.sym)
            self.syms = re.split(r'with|/', self.sym)

        self.sym_ents = [start_end_blank_pattern.sub('', sym) for sym in self.syms if start_end_blank_pattern.sub('', sym)]

        temp = []
        for word in self.loc.split(' '):
            if word not in preposition_loc:
                temp.append(word)
            elif temp:
                self.loc_ents.add(' '.join(temp))
                temp = []
        if temp:
            self.loc_ents.add(' '.join(temp))

    def num_mod2sym(self, residual, pair_sym, words):
        result = []
        sym_start = pair_sym[0]
        for item in residual:
            if ' '.join(words[item[0]:item[1]]) == 'change' or ' '.join(words[item[0]:item[1]]) == 'changes':
                continue
            if item[1] < sym_start and item[1] + 4 >= sym_start:
                result.append(' '.join(words[item[0]:item[1]]))
            if 'with' in set(words[i] for i in range(pair_sym[1], item[0])):
                result.append(' '.join(words[item[0]:item[1]]))
        return result
