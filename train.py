import tensorflow as tf
import os

import model_helper
import model
from utils import vocab_utils


def run_eval(eval_model, eval_sess, model_dir):
    with eval_model.graph.as_default():
        model_helper.load_model(eval_model.model, model_dir, eval_sess, name='eval')
        eval_sess.run(tf.tables_initializer())
    eval_model.model.eval(eval_sess, eval_model)


def train(num_train_steps=2, steps_per_eval=1):
    model_dir = '/tmp/rap_bot/model/'
    dataset_dir = '/tmp/rap_bot/dataset/'
    data_file = dataset_dir + 'data.lyric'
    label_file = dataset_dir + 'label.lyric'
    vocab_file = dataset_dir + 'lyrics.vocab'
    model_creator = model.Model
    train_model = model_helper.create_train_model(model_creator,
                                                  vocab_file,
                                                  data_file,
                                                  label_file)
    eval_model = model_helper.create_eval_model(model_creator,
                                                vocab_file)
    # Use in gpu.
    # config_proto = tf.ConfigProto(log_device_placement=True,
    #                               allow_soft_placement=True)
    # config_proto.gpu_options.allow_growth = True
    train_sess = tf.Session(graph=train_model.graph)
    eval_sess = tf.Session(graph=eval_model.graph)

    global_step = 0
    epoch_step = 0
    last_eval_step = global_step
    with train_model.graph.as_default():
        # Must init the two init op within the graph.
        train_sess.run(tf.global_variables_initializer())
        train_sess.run(tf.tables_initializer())
    train_sess.run(train_model.iterator.initializer)
    while global_step < num_train_steps:
        try:
            step_res = train_model.model.train(train_sess)
            print(step_res[1])
            epoch_step += 1
        except tf.errors.OutOfRangeError:
            epoch_step = 0
            train_sess.run(train_model.iterator.initializer)
            continue
        global_step = step_res[3]
        print(global_step)
        if global_step - last_eval_step > steps_per_eval:
            last_eval_step = global_step
            print('# Evaluating model and Save model at step {}.'.format(last_eval_step),
                  flush=True)
            train_model.model.saver.save(train_sess,
                                         os.path.join(model_dir, 'rap_bot.ckpt'),
                                         global_step=global_step)
            run_eval(eval_model, eval_sess, model_dir)




