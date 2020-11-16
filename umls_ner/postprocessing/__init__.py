import re
import csv
import pandas as pd
from collections import defaultdict
from .form_data import form_mammo_output, form_RE_data_by, form_NER_RE_data_by, form_negation_data_by

def row_merge_by(in_file, column_name, delimiter=None, remove_column=None):
    df = pd.read_csv(in_file, delimiter=delimiter, encoding='utf-8', dtype=str).fillna('')

    def app_func(df):
        info = []
        for col in df.columns:
            if col in column_name:
                info.append(df[col].iloc[0])
            else:
                ele_set = set([ele for field in df[col].values.tolist() for ele in field.split('|')])
                info.append('|'.join([str(ele) for ele in ele_set if ele]))
        return info

    result = [info for info in df.groupby(column_name).apply(app_func)]
    df_new = pd.DataFrame(result, columns=df.columns)
    if remove_column:
        df_new.drop(columns=remove_column, inplace=True)

    return df_new

def filter_loc_att(dataframe, loc_keys, att_patterns):

    def app_func(series):
        loc_set = set(series.Location.split('|'))
        speci_loc = loc_set & loc_keys
        if speci_loc:
            series['Location'] = '|'.join(speci_loc)
            att_set = set(att for att in series.Attribute.split('|') if any(pattern.search(att) for pattern in att_patterns))
            series['Attribute'] = '|'.join(att_set)
            return series.values.tolist()
        return None

    result = []
    infos = [info for info in dataframe.apply(app_func, axis=1) if info]

    for info in infos:
        for att in info[-2].split('|'):
            info[-2] = att
            result.append(list(info))

    df_new = pd.DataFrame(result, columns=dataframe.columns)

    return df_new

def field_normalization(map_file, dataframe, target):
    df_map = pd.read_csv(map_file, encoding='utf-8').fillna('')
    Term = df_map.Term.values.tolist()
    Synonymous = df_map.Synonymous.values.tolist()

    dict_map = defaultdict(str)
    for term, synonymous, in zip(Term, Synonymous):
        synonymous_list = synonymous.split('|')
        for vocab in synonymous_list:
            dict_map[vocab] = term

    Target = dataframe[target].values.tolist()
    for i, tar in enumerate(Target):
        if tar in dict_map:
            Target[i] = dict_map[tar]

    dataframe.loc[:,target] = Target

    return dataframe

def filter_sym(cancer_file, dataframe, out_file, mode, key='glioma', delimiter=None):
    # ====================
    # 病灶為:lesion
    # 且屬性為:nodular, patchy, solid, subsolid, hypodense, soft tissue, low density
    # ====================
    # 屬性有出現:nodular
    cancer_sym = set()
    content = open(cancer_file)
    lines = content.readlines()
    for line in lines:
        cancer_sym.add(line.strip('\n'))

    lesion_att_set = {'nodular', 'patchy', 'solid', 'subsolid', 'hypodense', 'soft tissue', 'low density'}

    def app_func_Lung_CT(series):
        
        if series.Detect != 'Yes':
            return None

        symptoms = set(series.Symptom.split('|'))
        attributes = set(series.Attribute.split('|'))

        if symptoms & cancer_sym or 'nodular' in attributes or ('lesion' in symptoms and attributes & lesion_att_set):
            return series.values.tolist()

        return None

    def app_func_Head_MRI(series):

        if series.Detect != 'Yes':
            return None
            
        symptoms = series.Symptom

        if key in symptoms:
            return series.values.tolist()
        
        return None

    if mode == 'Lung_CT':
        datas = [data for data in dataframe.apply(app_func_Lung_CT, axis=1) if data]
    elif mode == 'Head_MRI':
        datas = [data for data in dataframe.apply(app_func_Head_MRI, axis=1) if data]
    else:
        raise ValueError('mode must be Lung_CT or Head_MRI')

    foutput = open(out_file, 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(foutput, delimiter=delimiter)

    csv_writer.writerow(dataframe.columns)
    csv_writer.writerows(datas)

# ===== 合併同句子同病灶 =====
# df = row_merge_by(in_file='output/test_output.csv',
#                   column_name=['SentenceID', 'Text', 'Symptom', 'Detect'])

# ===== 病灶詞標準化 =====
# df_new = field_normalization(map_file='program_data/symptom_map_list.csv',
#                              dataframe=df,
#                              target='Symptom')

# ===== 篩選病灶詞 =====
# filter_sym(cancer_file='program_data/cancer_sym.txt',
#            dataframe=df_new,
#            out_file='output/test_output_final.csv',
#            mode='Lung_CT')
