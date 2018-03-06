import os
import codecs


def get_vocab_size(vocab_file):
    vocab_size = 0
    if os.path.exists(vocab_file):
        with codecs.getreader('utf-8')(open(vocab_file, 'rb')) as f:
            for word in f:
                vocab_size += 1
    return vocab_size
