import csv
import itertools
import numpy as np
import pandas as pd

def read_csv_xlsx(file_path, delimiter=None, dtype=None):
    if '.csv' in file_path:
        df = pd.read_csv(file_path, delimiter=delimiter, dtype=dtype, encoding='utf-8').fillna('')
    elif '.xlsx' in file_path:
        df = pd.read_excel(file_path).fillna('')
    else:
        raise ValueError('The type of the data file must be csv or xlsx.')
    return df

def filter_field(in_file, out_file, dict_field, delimiter=None, dtype=None):
    '''
    dict_field = {
                'CURE_PATH': ['U'], 
                'ORDER_CODE': ['33070B', '33071B', '33072B'], 
                'HOSP_ID_PKNO': ['71', '82', '13149', '26133', '21235']
    }
    '''
    df = read_csv_xlsx(in_file, delimiter=delimiter, dtype=dtype)

    for key, values in dict_field.items():
        df = df[df[key].isin(values)]

    df.to_csv(out_file, sep=delimiter, columns=df.columns, index=False)

def remove_dupl_ner_data(in_file, out_file):
    df = pd.read_csv(in_file)

    app_func = lambda df: [(word, tag) for word, tag in zip(df['Word'].values.tolist(),
                                                            df['Tag'].values.tolist())]
    grouped = df.groupby('Sentence #').apply(app_func)
    data = [([word for word, tag in sentence], [tag for word, tag in sentence]) for sentence in grouped]
    data.sort()
    new_data = list(k for k, _ in itertools.groupby(data))

    sentences = [word for (words, tags) in new_data for word in words]
    labels = [tag for (words, tags) in new_data for tag in tags]
    sent_ids = [i+1 for i, (words, tags) in enumerate(new_data) for word in words]

    ner_list = [sent_ids, sentences, labels]
    df_ner = pd.DataFrame(ner_list).transpose()
    df_ner.columns = df.columns

    df_ner.to_csv(out_file, index=False)

def split_ner_data(in_file, train_file, valid_file, test_file):
    df = pd.read_csv(in_file)

    num_sent = df.iloc[-1]['Sentence #']
    train_split = int(num_sent*0.7)
    valid_split = int(num_sent*0.8)

    df_train = df[df['Sentence #'] <= train_split]
    df_valid = df[(df['Sentence #'] > train_split) & (df['Sentence #'] <= valid_split)]
    df_test = df[df['Sentence #'] > valid_split]

    df_train.columns = df.columns
    df_valid.columns = df.columns
    df_test.columns = df.columns

    df_train.to_csv(train_file, index=False)
    df_valid.to_csv(valid_file, index=False)
    df_test.to_csv(test_file, index=False)

def remove_dupl_data(in_file, out_file):
    df = pd.read_csv(in_file)

    app_func = lambda series: series.values.tolist()
    data = [info for info in df.apply(app_func, axis=1)]

    data.sort()
    new_data = list(k for k, _ in itertools.groupby(data))

    df_re = pd.DataFrame(new_data)
    df_re.columns = df.columns

    df_re.to_csv(out_file, index=False)

def split_data(in_file, train_file, valid_file, test_file):
    df = pd.read_csv(in_file)
    total_data = df.values.tolist()
    num_data = len(total_data)
    index_array = np.arange(num_data)
    np.random.shuffle(index_array)
    total_data = [total_data[idx] for idx in index_array]

    train_split = int(num_data*0.7)
    valid_split = int(num_data*0.8)

    train_data = total_data[:train_split]
    valid_data = total_data[train_split:valid_split]
    test_data = total_data[valid_split:]

    df_train = pd.DataFrame(train_data)
    df_valid = pd.DataFrame(valid_data)
    df_test = pd.DataFrame(test_data)

    df_train.columns = df.columns
    df_valid.columns = df.columns
    df_test.columns = df.columns

    df_train.to_csv(train_file, index=False)
    df_valid.to_csv(valid_file, index=False)
    df_test.to_csv(test_file, index=False)
