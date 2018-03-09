import tensorflow as tf

import model_helper

__all__ = ['Model']


class Model(object):
    def __init__(self,
                 mode,
                 iterator,
                 vocab_table,
                 time_major,
                 vocab_size,
                 num_units=256,
                 learning_rate=.1,
                 batch_size=32,
                 max_gradient_norm=5.,
                 num_ckpts_to_keep=10,
                 reverse_vocab_table=None,
                 inputs=None):
        self.num_units = num_units
        self.mode = mode
        self.iterator = iterator
        self.vocab_table = vocab_table
        self.time_major = time_major
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.max_gradient_norm = max_gradient_norm
        # Use in eval.
        if inputs is not None:
            self.inputs = inputs
        self.embedding_matrix = tf.Variable(tf.one_hot(list(range(vocab_size)), depth=self.vocab_size),
                                            trainable=False)
        with tf.variable_scope('build_network'):
            with tf.variable_scope('output_projection'):
                self.output_layer = tf.layers.Dense(
                    self.vocab_size, use_bias=False, name='output_projection'
                )
        res = self.build_graph()
        self.outputs = res[0]
        self.state = res[1]
        self.logits = res[2]
        self.probas = tf.nn.softmax(self.logits)
        self.output_ids = tf.argmax(self.probas, axis=2)
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        params = tf.trainable_variables()
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.train_loss = res[3]
            self.learning_rate = tf.constant(learning_rate)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            gradients = tf.gradients(self.train_loss, params)
            clipped_grads, gradient_norm = tf.clip_by_global_norm(gradients,
                                                                  self.max_gradient_norm)
            self.update = optimizer.apply_gradients(zip(clipped_grads, params),
                                                    global_step=self.global_step)
        else:
            self.output_words = reverse_vocab_table.lookup(tf.to_int64(self.output_ids))

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_ckpts_to_keep)
        print('# Trainable variables')
        for param in params:
            print('  {}, {}'.format(param.name, str(param.get_shape())))

    def build_graph(self):
        print('# Building {} graph ...'.format(self.mode), flush=True)
        dtype = tf.float32
        with tf.variable_scope('rap_rnn', dtype=dtype):
            iterator = self.iterator
            if iterator is not None:
                # inputs is batch_size major
                self.inputs = iterator.inputs
            if self.time_major:
                inputs = tf.transpose(self.inputs)
            with tf.variable_scope('rnn'):
                emb_inp = tf.nn.embedding_lookup(self.embedding_matrix,
                                                 inputs)
                print('  num layers of rnn: {}'.format(1))
                cell = model_helper.create_rnn_cell(cell_type='lstm',
                                                    num_units=self.num_units)
                self.initial_state = cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
                outputs, state = tf.nn.dynamic_rnn(cell, emb_inp,
                                                   dtype=dtype,
                                                   time_major=self.time_major,
                                                   swap_memory=True,
                                                   initial_state=self.initial_state)
                logits = self.output_layer(outputs)
                # Doesn't compute loss in eval in here.
                if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
                    loss = self._compute_loss(logits)
                    return outputs, state, logits, loss
                return outputs, state, logits

    def _compute_loss(self, logits):
        labels = self.iterator.outputs
        if self.time_major:
            labels = tf.transpose(labels)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits
        )
        loss = tf.reduce_sum(cross_entropy) / tf.to_float(self.batch_size)
        return loss

    def train(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
        return sess.run([self.logits, self.train_loss, self.update, self.global_step])

    def eval(self, sess, eval_model):
        assert self.mode == tf.contrib.learn.ModeKeys.EVAL
        print('--------', sess.run(self.inputs, feed_dict={eval_model.input_placeholder: '必 须'}))
        all_output_ids = []
        probas, state, output_id, output_word = sess.run([self.probas, self.state, self.output_ids, self.output_words],
                                             feed_dict={
                                                 eval_model.input_placeholder: '必 须'
                                             })
        time_steps = output_id.shape[0]
        for time_step in range(time_steps):
            all_output_ids.append(output_word[time_step])
        while time_steps < 20:
            (proba, state, output_word) = sess.run([self.probas, self.state, self.output_words],
                                        feed_dict={eval_model.input_placeholder: output_word[-1][-1],
                                                   self.initial_state: state})
            all_output_ids.append(output_word[0])
            print('time_steps {}'.format(time_steps))
            time_steps += 1
        print('time_steps', time_steps)
        print(all_output_ids)
