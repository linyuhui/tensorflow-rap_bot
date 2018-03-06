import codecs
import os

from utils import spider_utils
from utils import char_utils


def clean_lrc(lrc, min_len=6):
    lrc = lrc[10:].strip()
    if len(lrc) < min_len:
        return None
    cleaned_lrc = ''
    for char in lrc:
        if char_utils.is_chinese(char):
            cleaned_lrc += char
    # After deleting illegal characters,
    # the length may be less than minimum length.
    if len(cleaned_lrc) < min_len:
        return None
    cleaned_lrc += '\n'
    return cleaned_lrc


def handle_lyric(lyric, f):
    # Skip lines of '作曲' and '作词'.
    lyric = lyric.split('\n')[2:]
    for lrc in lyric:
        cleaned_lrc = clean_lrc(lrc)
        if cleaned_lrc:
            f.write(cleaned_lrc)
        else:
            pass


def prepare_lyrics(out_file):
    playlist_info = spider_utils.get_playlist('说唱', offset=0)
    playlist_id = playlist_info.playlist_id[
        playlist_info.playlist_name.index('中国饶舌')
    ]
    music_info = spider_utils.get_music(playlist_id)
    f = codecs.getwriter('utf-8')(open(out_file, 'wb'))
    print('Starting handle lyrics. \n There are {} songs in all.'.format(
        len(music_info.music_id)
    ))
    for music_id in music_info.music_id:
        lyric = spider_utils.get_lyric(music_id)
        handle_lyric(lyric, f)
    f.close()
    print(' Done.')


def build_vocab_file(in_file, out_file, use_words=20000):
    with codecs.getreader('utf-8')(open(in_file, 'rb')) as f:
        corpus_chars = f.read()
    print(len(corpus_chars))
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[:use_words]
    unique_chars = list(set(corpus_chars))
    print('Starting building vocabulary file {}.'.format(out_file))
    vocab_size = len(unique_chars)
    print(' vocab size:', vocab_size)
    with codecs.getwriter('utf-8')(open(out_file, 'wb')) as f:
        for char in unique_chars:
            f.write(char + '\n')
    print(' Done.')


def generate_examples_files(in_file, out_suffix, data_prefix, label_prefix,
                            time_steps, use_words=20000):
    print(' Starting generating examples ...')
    data_file = os.path.join(os.path.dirname(in_file),
                             data_prefix + out_suffix)
    label_file = os.path.join(os.path.dirname(in_file),
                              label_prefix + out_suffix)
    with codecs.getreader('utf-8')(open(in_file, 'rb')) as f:
        corpus_chars = f.read()
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[:use_words]
    corpus_chars = list(corpus_chars)
    # Index of label is index of data plus one.
    num_examples = (len(corpus_chars) - 1) // time_steps
    example_indices = list(range(num_examples))
    f_data = codecs.getwriter('utf-8')(open(data_file, 'wb'))
    f_label = codecs.getwriter('utf-8')(open(label_file, 'wb'))

    def _example(pos):
        return corpus_chars[pos:pos + time_steps]

    for index in example_indices:
        feature = _example(index * time_steps)
        feature = '/'.join(feature)
        label = _example(index * time_steps + 1)
        label = '/'.join(label)
        f_data.write(feature + '\n')
        f_label.write(label + '\n')
    f_data.close()
    f_label.close()
    print('  Done.')


if __name__ == '__main__':
    dataset_dir = '/tmp/rap_bot/dataset/'
    lyric_filename = 'rap_lyrics.txt'
    vocab_filename = 'lyrics.vocab'
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    prepare_lyrics(os.path.join(dataset_dir, lyric_filename))

    build_vocab_file(os.path.join(dataset_dir, lyric_filename),
                     os.path.join(dataset_dir, vocab_filename))
    generate_examples_files(os.path.join(dataset_dir, lyric_filename),
                            '.lyric',
                            data_prefix='data',
                            label_prefix='label',
                            time_steps=30)
