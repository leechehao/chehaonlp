from keras.preprocessing.sequence import pad_sequences


class DataStream(object):
    def __init__(self, dataset, tokenizer, FLAGS, mode):
        self.dataset = dataset
        self.tokenizer = tokenizer
        if mode == 're':
            app_func = lambda row: [row['Symptom'], row['Target'], row['Text'], row['Relation']]
            self.sentences = [(sym, tar, txt) for sym, tar, txt, tag in self.dataset.apply(app_func, axis=1)]
            self.labels = [FLAGS['tag2idx'][tag] for sym, tar, txt, tag in self.dataset.apply(app_func, axis=1)]
            tokenized_texts_and_type_ids = [self.tokenized_and_special_token(sym, txt, tar) for sym, tar, txt in self.sentences]
        elif mode == 'neges':
            app_func = lambda row: [row['Symptom'], row['Text'], row['Detect']]
            self.sentences = [(sym, txt) for sym, txt, tag in self.dataset.apply(app_func, axis=1)]
            self.labels = [FLAGS['tag2idx'][tag] for sym, txt, tag in self.dataset.apply(app_func, axis=1)]
            tokenized_texts_and_type_ids = [self.tokenized_and_special_token(sym, txt) for sym, txt in self.sentences]
        else:
            raise ValueError('Please give correct DataStream mode !!!')

        self.tokenized_sentences = [token_type_pair[0] for token_type_pair in tokenized_texts_and_type_ids]
        token_type_ids = [token_type_pair[1] for token_type_pair in tokenized_texts_and_type_ids]

        self.input_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(sent) for sent in self.tokenized_sentences],
                                        maxlen=FLAGS['MAX_LEN'], dtype='int64', value=0.0,
                                        padding='post', truncating='post')

        self.token_type_ids = pad_sequences(token_type_ids,
                                            maxlen=FLAGS['MAX_LEN'], dtype='int64', value=0.0,
                                            padding='post', truncating='post')

        self.attention_masks = [[int(i != 0.0) for i in ids] for ids in self.input_ids]

    def tokenized_and_special_token(self, sym, txt, *args):
        tokenized_sym = self.tokenizer.tokenize(sym)+[self.tokenizer.sep_token]
        tokenized_txt = self.tokenizer.tokenize(txt)+[self.tokenizer.sep_token]

        tokenized_args = []
        for arg in args:
            tokenized_args.extend(self.tokenizer.tokenize(arg)+[self.tokenizer.sep_token])

        tokenized_sentence = [self.tokenizer.cls_token] + tokenized_sym + tokenized_args + tokenized_txt
        token_type_id = [0] + [0]*len(tokenized_sym+tokenized_args) + [1]*len(tokenized_txt)

        return tokenized_sentence, token_type_id
