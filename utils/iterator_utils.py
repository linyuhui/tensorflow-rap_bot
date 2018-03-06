import tensorflow as tf
import collections


class BatchedInput(collections.namedtuple('BatchedInput',
                                          ('initializer', 'inputs', 'outputs'))):
    pass


def get_iterator(data_dataset, label_dataset, vocab_table, batch_size=32):
    dataset = tf.data.Dataset.zip((data_dataset, label_dataset))
    buffer_size = 10 * batch_size
    dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)
    dataset = dataset.map(lambda data, label: (
        tf.string_split([data], delimiter='/').values,
        tf.string_split([label], delimiter='/').values,
    ), num_parallel_calls=4)
    dataset = dataset.map(lambda data, label: (tf.cast(vocab_table.lookup(data), tf.int32),
                                               tf.cast(vocab_table.lookup(label), tf.int32)),
                          num_parallel_calls=4)
    batched_dataset = dataset.batch(batch_size)
    batched_iter = batched_dataset.make_initializable_iterator()
    input_ids, output_ids = batched_iter.get_next()
    return BatchedInput(initializer=batched_iter.initializer,
                        inputs=input_ids,
                        outputs=output_ids)


def get_inputs(input_placeholder, vocab_table):
    inputs = tf.string_split([input_placeholder], delimiter=' ').values
    input_ids = tf.cast(vocab_table.lookup(inputs), tf.int32)
    input_ids = tf.reshape(input_ids, shape=[1, -1])
    return input_ids
