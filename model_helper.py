import tensorflow as tf
import collections

from utils import iterator_utils
from utils import vocab_utils


class TrainModel(collections.namedtuple('TrainModel',
                                        ('graph', 'model', 'iterator'))):
    pass


class EvalModel(collections.namedtuple('EvalModel',
                                       ('graph', 'model', 'input_placeholder'))):
    pass


def create_train_model(model_creator, vocab_file, data_file, label_file):
    graph = tf.Graph()
    with graph.as_default(), tf.container('train'):
        vocab_table = tf.contrib.lookup.index_table_from_file(vocab_file,
                                                              name='vocab_table',
                                                              default_value=0)
        data_dataset = tf.data.TextLineDataset(data_file)
        label_dataset = tf.data.TextLineDataset(label_file)
        iterator = iterator_utils.get_iterator(data_dataset, label_dataset, vocab_table,
                                               batch_size=32)
        vocab_size = vocab_utils.get_vocab_size(vocab_file)
        with tf.device('/cpu:0'):
            model = model_creator(
                mode=tf.contrib.learn.ModeKeys.TRAIN,
                iterator=iterator,
                vocab_table=vocab_table,
                vocab_size=vocab_size,
                time_major=True,
                num_units=256
            )
    return TrainModel(graph=graph,
                      model=model,
                      iterator=iterator)


def create_eval_model(model_creator, vocab_file):
    graph = tf.Graph()
    with graph.as_default(), tf.container('eval'):
        vocab_table = tf.contrib.lookup.index_table_from_file(vocab_file,
                                                              name='vocab_table',
                                                              default_value=0)
        reverse_vocab_table = tf.contrib.lookup.index_to_string_table_from_file(vocab_file,
                                                                                default_value=' ')
        input_placeholder = tf.placeholder(dtype=tf.string, shape=None)
        inputs = iterator_utils.get_inputs(input_placeholder, vocab_table)
        vocab_size = vocab_utils.get_vocab_size(vocab_file)
        with tf.device('/cpu:0'):
            model = model_creator(mode=tf.contrib.learn.ModeKeys.EVAL,
                                  iterator=None,
                                  vocab_table=vocab_table,
                                  vocab_size=vocab_size,
                                  time_major=True,
                                  num_units=256,
                                  learning_rate=.1,
                                  batch_size=1,
                                  max_gradient_norm=5.,
                                  num_ckpts_to_keep=10,
                                  inputs=inputs,
                                  reverse_vocab_table=reverse_vocab_table)
        return EvalModel(graph=graph,
                         model=model,
                         input_placeholder=input_placeholder)


def create_rnn_cell(cell_type, num_units, dropout=0., device='/cpu:0', forget_bias=1.):
    if cell_type == 'lstm':
        print('  LSTM, forget_bias: {}'.format(forget_bias),
              flush=True, end='')
        cell = tf.nn.rnn_cell.LSTMCell(num_units,
                                       forget_bias=forget_bias)
    elif cell_type == 'gru':
        print('  GRU', flush=True, end='')
        cell = tf.nn.rnn_cell.GRUCell(num_units)
    if dropout > 0.:
        cell = tf.contrib.rnn.DropoutWrapper(cell=cell,
                                             input_keep_prob=(1. - dropout))
        print('  dropout: {}'.format(dropout), end='')
    if device:
        cell = tf.contrib.rnn.DeviceWrapper(cell, device)
        print(' {}, device: {}'.format(type(cell).__name__, device), end='',
              flush=True)
    print('', flush=True)
    return cell


def load_model(model, model_dir, session, name):
    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    model.saver.restore(session, latest_ckpt)
    print('  loaded {} parameters from {}'.format(name, latest_ckpt))