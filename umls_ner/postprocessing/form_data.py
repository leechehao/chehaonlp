import re
import csv
import pandas as pd

def form_mammo_output(in_file, out_file):
    df = pd.read_csv(in_file).fillna('')

    mammo_loc_set = {'UOQ', 'UIQ', 'LOQ', 'LIQ', 'subareolar', 'Subareolar', 'axilla', 'Axilla'}
    up_low_patt = re.compile('upper|lower', re.IGNORECASE)
    in_out_patt = re.compile('inner|outer', re.IGNORECASE)

    right_patt = re.compile('right', re.IGNORECASE)
    left_patt = re.compile('left', re.IGNORECASE)

    def app_func(series):
        locations = series.Location
        qualified = False
        side = ''
        note = ''
        
        if series.Detect != 'Yes':
            return series.values.tolist()+[side, note]

        if locations:
            if any(word in locations for word in mammo_loc_set) or (up_low_patt.search(locations) and in_out_patt.search(locations)):
                qualified = True
            if right_patt.search(locations):
                side = 'right'
            elif left_patt.search(locations):
                side = 'left'
            else:
                side = 'bilateral'

        if qualified:
            return series.values.tolist()+[side, note]
        else:
            note = series.loc['Location']
            series.loc['Location'] = ''
            return series.values.tolist()+[side, note]

    new_data = list(df.apply(app_func, axis=1))
    pd.DataFrame(new_data, columns=list(df.columns)+['Side', 'Note']).to_csv(out_file, index=False)

def form_RE_data_by(in_file, out_file, column_name):

    def app_func(df):
        yes_pair, no_pair = [], []
        all_sym, all_loc, all_att = set(), set(), set()

        text = df.Text.iloc[0]
        sym_fields = df.Symptom.values.tolist()
        loc_fields = df.Location.values.tolist()
        att_fields = df.Attribute.values.tolist()

        for sym_field, loc_field, att_field in zip(sym_fields, loc_fields, att_fields):
            locs = set(loc for loc in loc_field.split('|') if loc)
            atts = set(att for att in att_field.split('|') if att)
            all_loc = all_loc | locs
            all_att = all_att | atts
            if sym_field:
                all_sym.add(sym_field)
                for loc in locs:
                    yes_pair.append((sym_field, loc))
                for att in atts:
                    yes_pair.append((sym_field, att))

        for sym in all_sym:
            for loc in all_loc:
                if (sym, loc) not in yes_pair:
                    no_pair.append((sym, loc))
            for att in all_att:
                if (sym, att) not in yes_pair:
                    no_pair.append((sym, att))
        info = [text, yes_pair, no_pair]
        return info

    df = pd.read_csv(in_file, encoding='utf-8').fillna('')
    df_grouped = [info for info in df.groupby(column_name).apply(app_func)]

    foutput = open(out_file, 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(foutput)
    csv_writer.writerow(['Symptom', 'Target', 'Text', 'Relation'])

    for text, yes_pair, no_pair in df_grouped:
        content = []
        for pair in yes_pair:
            content.append([pair[0], pair[1], text, 'Yes'])
        for pair in no_pair:
            content.append([pair[0], pair[1], text, 'No'])

        csv_writer.writerows(content)

def form_NER_RE_data_by(in_file, out_file, column_name):

    def app_func(df):
        text = df.Text.iloc[0]
        sym_fields = df.Symptom.values.tolist() # ['(3, 4)', '(3, 4)']
        tar_fields = df.Target.values.tolist() # ['(4, 5)', '(5, 6)']
        rel_fields = df.Relation.values.tolist() # ['Yes', 'Yes']

        span1s = []
        span2s = []
        mentions = set()

        for sym, tar, rel in zip(sym_fields, tar_fields, rel_fields):
            sym = tuple(map(int, re.findall(r'\d+', sym)))
            tar = tuple(map(int, re.findall(r'\d+', tar)))
            if rel == 'Yes':
                span_pair = sorted([sym, tar], key=lambda x: x[0])
                span1s.append(span_pair[0])
                span2s.append(span_pair[1])
            mentions.add(sym)
            mentions.add(tar)

        return [text, span1s, span2s, list(mentions)]

    df = pd.read_csv(in_file, encoding='utf-8').fillna('')
    df_grouped = [info for info in df.groupby(column_name).apply(app_func)]

    new_df = pd.DataFrame(df_grouped, columns=['Text', 'Span1', 'Span2', 'Mention'])
    new_df.to_csv(out_file, index=False)

def form_negation_data_by(in_file, out_file):
    df = pd.read_csv(in_file, encoding='utf-8').fillna('')

    df_sym = df[df['Detect'] != '']
    neg_list = [df_sym.Symptom.tolist(), df_sym.Text.tolist(), df_sym.Detect.tolist()]
    df_neg = pd.DataFrame(neg_list).transpose()
    df_neg.columns = ['Symptom', 'Text', 'Detect']

    df_neg.to_csv(out_file, index=False)

# ===== 生成乳房輸出格式 =====
# form_mammo_output(in_file='output/test_output.csv',
#                   out_file='output/test_output_mammo.csv')

# ===== 生成 Relation Extraction 資料 =====
# form_RE_data_by(in_file='output/test_output.csv',
#                 out_file='output/test_pair.csv',
#                 column_name=['SentenceID', 'Text'])

# # ===== 生成 NER-RE Span 資料 =====
# form_NER_RE_data_by(in_file='output/test_span_pair.csv',
#                     out_file='output/test_ner-re.csv',
#                     column_name=['Text'])

# # ===== 生成 病灶否定 資料 =====
# form_negation_data_by(in_file='output/test_output.csv',
#                       out_file='output/test_neg.csv')