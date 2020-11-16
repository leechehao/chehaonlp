from keras.preprocessing.sequence import pad_sequences


class DataStream(object):
    def __init__(self, dataset, tokenizer, FLAGS):
        self.dataset = dataset
        self.tokenizer = tokenizer
        app_func = lambda df: [(word, tag) for word, tag in zip(df['Word'].values.tolist(),
                                                                df['Tag'].values.tolist())]
        self.grouped = self.dataset.groupby('Sentence #').apply(app_func)
        self.sentences = [[word for word, tag in sentence] for sentence in self.grouped]
        self.labels = [[tag for word, tag in sentence] for sentence in self.grouped]

        tokenized_sentences_and_labels = [
            self.tokenize_and_preserve_tags(sent, tags)
            for sent, tags in zip(self.sentences, self.labels)
        ]
        self.tokenized_sentences = [data_pair[0] for data_pair in tokenized_sentences_and_labels]
        self.tokenized_labels = [data_pair[1] for data_pair in tokenized_sentences_and_labels]

        self.input_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(sent) for sent in self.tokenized_sentences],
                                        maxlen=FLAGS['MAX_LEN'], dtype='int64', value=0.0,
                                        padding='post', truncating='post')

        self.labels = pad_sequences([[FLAGS['tag2idx'].get(tag) for tag in label] for label in self.tokenized_labels],
                                    maxlen=FLAGS['MAX_LEN'], dtype='int64', value=FLAGS['tag2idx']['PAD'],
                                    padding='post', truncating='post')

        self.attention_masks = [[int(i != 0) for i in ids] for ids in self.input_ids]

    def tokenize_and_preserve_tags(self, sentence, tags):
        tokenized_sentence = []
        tokenized_tags = []

        for word, tag in zip(sentence, tags):
            # Tokenize the word and count # of subwords the word is broken into
            subwords = self.tokenizer.tokenize(word)
            n_subwords = len(subwords)

            # Add the tokenized word to the final tokenized word list
            tokenized_sentence.extend(subwords)
            
            # Add the same label to the new list of tags `n_subwords` times
            tokenized_tags.extend([tag] * n_subwords)

        return tokenized_sentence, tokenized_tags
