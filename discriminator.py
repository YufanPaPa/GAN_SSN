import tensorflow as tf
import numpy as np
import os
import time
import random
from six.moves import xrange
from model.dis_model import Hier_rnn_model
import util

from tensorflow.python.platform import gfile
import sys


def hier_read_data(config, query_path, answer_path, gen_path):
    query_set = [[] for _ in config.buckets]
    answer_set = [[] for _ in config.buckets]
    gen_set = [[] for _ in config.buckets]
    with gfile.GFile(query_path, mode="r") as query_file:
        with gfile.GFile(answer_path, mode="r") as answer_file:
            with gfile.GFile(gen_path, mode="r") as gen_file:
                query, answer, gen = query_file.readline(), answer_file.readline(), gen_file.readline()
                counter = 0
                while query and answer and gen:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  reading disc_data line %d" % counter)
                    query = [int(id) for id in query.strip().split()]
                    answer = [int(id) for id in answer.strip().split()]
                    gen = [int(id) for id in gen.strip().split()]
                    for i, (query_size, answer_size) in enumerate(config.buckets):
                        if len(query) <= query_size and len(answer) <= answer_size and len(gen) <= answer_size:
                            query = query[:query_size] + [util.PAD_ID] * (
                                query_size - len(query) if query_size > len(query) else 0)
                            query_set[i].append(query)
                            answer = answer[:answer_size] + [util.PAD_ID] * (
                                answer_size - len(answer) if answer_size > len(answer) else 0)
                            answer_set[i].append(answer)
                            gen = gen[:answer_size] + [util.PAD_ID] * (
                                answer_size - len(gen) if answer_size > len(gen) else 0)
                            gen_set[i].append(gen)
                            break
                    query, answer, gen = query_file.readline(), answer_file.readline(), gen_file.readline()

    return query_set, answer_set, gen_set


def hier_get_batch(config, max_set, query_set, answer_set, gen_set):
    batch_size = config.batch_size
    if batch_size % 2 == 1:
        return IOError("Error")
    train_query = []
    train_answer = []
    train_labels = []
    half_size = int(batch_size / 2)
    for _ in range(half_size):
        index = random.randint(0, max_set)
        train_query.append(query_set[index])
        train_answer.append(answer_set[index])
        train_labels.append(1)
        train_query.append(query_set[index])
        train_answer.append(gen_set[index])
        train_labels.append(0)
    return train_query, train_answer, train_labels


def get_cur_sen_slices(config, sen_list):
    query_slice_set = []
    cur_query_slice = []
    for item in sen_list:
        if item != 6:
            cur_query_slice.append(item)
        else:
            if len(cur_query_slice) != 0:
                query_slice_set.append(cur_query_slice)
                cur_query_slice = []
    if len(cur_query_slice) != 0:
        query_slice_set.append(cur_query_slice)
    current_query = [num for num in query_slice_set[-1] if num != 0]
    query_answer_pair_set = []
    if len(query_slice_set) >= 2:
        for idx in range(len(query_slice_set) - 1):
            cur_query = [num for num in query_slice_set[idx] if num != 0]
            cur_response = [num for num in query_slice_set[idx + 1] if num != 0]
            cur_qa = cur_query + [6] + cur_response
            cur_qa = cur_qa[:2 * config.max_len] + [util.PAD_ID] * (
                2 * config.max_len - len(cur_qa) if 2 * config.max_len > len(cur_qa) else 0)
            query_answer_pair_set.append(cur_qa)

    return query_answer_pair_set, current_query


def get_random_qa(config, query_slice_set, current_query, answer, generation):
    # query_answer_set is a list of q-a sequence
    # we need to get context_qa_order[3,seq_length],context_qa_disorder[3,seq_length],current_qa_disorder[3,seq_length]
    order_sequence = []
    disorder_sequence = []
    current_sequence_positive = []
    current_sequence_negative = []
    current_qa_positive = current_query + [6] + answer
    current_qa_positive = current_qa_positive[:2 * config.max_len] + [util.PAD_ID] * (
        2 * config.max_len - len(current_qa_positive) if 2 * config.max_len > len(current_qa_positive) else 0)
    current_qa_negative = current_query + [6] + generation
    current_qa_negative = current_qa_negative[:2 * config.max_len] + [util.PAD_ID] * (
        2 * config.max_len - len(current_qa_negative) if 2 * config.max_len > len(current_qa_negative) else 0)
    padding_sequence = [7 for i in range(2*config.max_len)]
    while len(query_slice_set) < 4:
        query_slice_set.insert(0,padding_sequence)

    if len(query_slice_set) >= 4:
        start_idx = random.randint(0, len(query_slice_set) - 3)
        order_sequence.append(query_slice_set[start_idx])
        order_sequence.append(query_slice_set[start_idx + 1])
        order_sequence.append(query_slice_set[start_idx + 2])

        start_idx = random.randint(0, len(query_slice_set) - 3)
        disorder_sequence.append(query_slice_set[start_idx + 1])
        disorder_sequence.append(query_slice_set[start_idx + 2])
        disorder_sequence.append(query_slice_set[start_idx])

        start_idx = random.randint(0, len(query_slice_set) - 2)
        current_sequence_positive.append(query_slice_set[start_idx])
        current_sequence_positive.append(current_qa_positive)
        current_sequence_positive.append(query_slice_set[start_idx + 1])

        current_sequence_negative.append(query_slice_set[start_idx])
        current_sequence_negative.append(current_qa_negative)
        current_sequence_negative.append(query_slice_set[start_idx + 1])

        return order_sequence, disorder_sequence, current_sequence_positive, current_sequence_negative



def hier_get_qa_batch(config, max_set, query_set, answer_set, gen_set):
    batch_size = config.batch_size
    if batch_size % 2 == 1:
        return IOError("Error")
    context_order = []
    context_unorder = []
    current_sen = []
    train_labels = []
    half_size = int(batch_size / 2)
    for _ in range(half_size):
        index = random.randint(0, max_set)
        query_answer_set, current_query = get_cur_sen_slices(config, query_set[index])
        context_positive, context_negative, current_qa_positive, current_qa_negative = get_random_qa(config,
                                                                                                     query_answer_set,
                                                                                                     current_query,
                                                                                                     answer_set[index],
                                                                                                     gen_set[index])
        context_order.append(context_positive)
        context_unorder.append(context_negative)
        current_sen.append(current_qa_positive)
        train_labels.append(1)

        context_order.append(context_positive)
        context_unorder.append(context_negative)
        current_sen.append(current_qa_negative)
        train_labels.append(0)

    return context_order, context_unorder, current_sen, train_labels


def create_model(sess, config, name_scope, initializer=None):
    with tf.variable_scope(name_or_scope=name_scope, initializer=initializer):
        model = Hier_rnn_model(config=config, name_scope=name_scope)
        disc_ckpt_dir = config.checkpoint_dir
        ckpt = tf.train.get_checkpoint_state(disc_ckpt_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading Hier Disc model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Created Hier Disc model with fresh parameters.")
            disc_global_variables = [gv for gv in tf.global_variables() if name_scope in gv.name]
            sess.run(tf.variables_initializer(disc_global_variables))
        return model


def prepare_data(config):
    vocab_path = config.vocab_dir
    vocab, rev_vocab = util.initialize_vocabulary(vocab_path)
    # vocab: a dict, mapping word to id
    # rev_vocab: a list, mapping id to word

    print("Preparing train disc_data in %s" % config.train_dir)
    train_query_path, train_answer_path, train_gen_path, dev_query_path, dev_answer_path, dev_gen_path = \
        util.hier_prepare_disc_data(config.train_dir, config.dev_dir, vocab, config.vocab_size)
    query_set_train, answer_set_train, gen_set_train = hier_read_data(config, train_query_path, train_answer_path,
                                                                      train_gen_path)
    query_set_dev, answer_set_dev, gen_set_dev = hier_read_data(config, dev_query_path, dev_answer_path, dev_gen_path)
    return query_set_train, answer_set_train, gen_set_train, query_set_dev, answer_set_dev, gen_set_dev


def softmax(x):
    prob = np.exp(x) / np.sum(np.exp(x), axis=0)
    return prob
