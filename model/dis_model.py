import tensorflow as tf
import numpy as np
from six.moves import xrange


def embedding(inputs,
              vocab_size,
              num_units,
              zero_pad=True,
              scale=True,
              scope="embedding",
              reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * (num_units ** 0.5)

    return outputs


class Hier_rnn_model(object):
    def __init__(self, config, name_scope, dtype=tf.float32):
        emb_dim = config.embed_dim
        vocab_size = config.vocab_size
        num_class = config.num_class
        self.lr = config.lr
        self.global_step = tf.Variable(initial_value=0, trainable=False)
        self.max_qa_len = 2 * config.max_len

        self.context_order = tf.placeholder(dtype=tf.int32, shape=[None, 3, self.max_qa_len], name="context_order")
        self.context_unorder = tf.placeholder(dtype=tf.int32, shape=[None, 3, self.max_qa_len], name="context_unorder")
        self.current_unorder = tf.placeholder(dtype=tf.int32, shape=[None, 3, self.max_qa_len], name="current_qa")

        context_order = tf.reshape(self.context_order, [-1, self.max_qa_len])
        context_unorder = tf.reshape(self.context_unorder, [-1, self.max_qa_len])
        current_unorder = tf.reshape(self.current_unorder, [-1, self.max_qa_len])

        context_order_mask = tf.cast(tf.cast(context_order, tf.bool), tf.float32)
        context_order_length = tf.reduce_sum(tf.cast(context_order_mask, tf.int32), axis=1)

        context_unorder_mask = tf.cast(tf.cast(context_unorder, tf.bool), tf.float32)
        context_unorder_length = tf.reduce_sum(tf.cast(context_unorder_mask, tf.int32), axis=1)

        current_unorder_mask = tf.cast(tf.cast(current_unorder, tf.bool), tf.float32)
        current_unorder_length = tf.reduce_sum(tf.cast(current_unorder_mask, tf.int32), axis=1)

        self.target = tf.placeholder(dtype=tf.int64, shape=[None], name="target")

        context_order_emb = embedding(context_order, vocab_size, emb_dim, scope='encoder_emb', reuse=tf.AUTO_REUSE)
        context_unorder_emb = embedding(context_unorder, vocab_size, emb_dim, scope='encoder_emb',
                                        reuse=tf.AUTO_REUSE)
        current_unorder_emb = embedding(current_unorder, vocab_size, emb_dim, scope='encoder_emb',
                                        reuse=tf.AUTO_REUSE)

        self.qa_encoder = []
        self.b_answer_state = []
        self.b_state = []
        self.b_logits = []
        self.b_loss = []
        self.b_train_op = []
        for (emb, length) in zip([context_order_emb, context_unorder_emb, current_unorder_emb],
                                      [context_order_length, context_unorder_length,
                                       current_unorder_length]):
            with tf.variable_scope(name_or_scope="Utterance_Pair_Encoder", reuse=tf.AUTO_REUSE) as var_scope:
                fw_cell = tf.nn.rnn_cell.GRUCell(emb_dim / 2)
                bw_cell = tf.nn.rnn_cell.GRUCell(emb_dim / 2)
                outputs, (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
                    fw_cell,
                    bw_cell,
                    emb,
                    sequence_length=length,
                    dtype=tf.float32
                )
                fw_state = tf.reshape(fw_state,[-1,3,int(emb_dim/2)])
                bw_state = tf.reshape(bw_state, [-1, 3, int(emb_dim / 2)])
                cur_emb_encoder = tf.concat([fw_state, bw_state], 2)
                self.qa_encoder.append(cur_emb_encoder)
        # qa_encoder = tf.stack(self.qa_encoder, axis=1)
        self.reason_encoder = []
        for encoder_vec in self.qa_encoder:
            with tf.variable_scope(name_or_scope="Qrder_Reasoning_Layer",reuse=tf.AUTO_REUSE) as var_scope:
                fw_cell = tf.nn.rnn_cell.GRUCell(emb_dim / 2)
                bw_cell = tf.nn.rnn_cell.GRUCell(emb_dim / 2)
                outputs, (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
                    fw_cell,
                    bw_cell,
                    encoder_vec,
                    dtype=tf.float32
                )
                # fw_output = tf.reshape(outputs[0], [config.batch_size, 3, int(emb_dim / 2), 1])
                # bw_output = tf.reshape(outputs[1], [config.batch_size, 3, int(emb_dim / 2), 1])
                # fw_encoder = tf.reshape(tf.nn.max_pool(fw_output, [1, 3, 1, 1], [1, 1, 1, 1], 'VALID'),
                #                         [config.batch_size, int(emb_dim / 2)])
                # bw_encoder = tf.reshape(tf.nn.max_pool(bw_output, [1, 3, 1, 1], [1, 1, 1, 1], 'VALID'),
                #                         [config.batch_size, int(emb_dim / 2)])
                fw_encoder = tf.reduce_sum(outputs[0],axis=1)
                bw_encoder = tf.reduce_sum(outputs[1],axis=1)
                final_encoder = tf.concat([fw_encoder, bw_encoder], axis=1)
                self.reason_encoder.append(final_encoder)
        final_sample_encoder = tf.reshape(tf.stack(self.reason_encoder, axis=1),[-1,emb_dim*3])

        with tf.variable_scope("Softmax_layer_and_output", reuse=tf.AUTO_REUSE):
            softmax_w = tf.get_variable("softmax_w", [3*emb_dim, num_class], dtype=tf.float32)
            softmax_b = tf.get_variable("softmax_b", [num_class], dtype=tf.float32)
            logits = tf.matmul(final_sample_encoder, softmax_w) + softmax_b
            self.b_logits.append(logits)

        with tf.name_scope("loss"):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.target)
            mean_loss = tf.reduce_mean(loss)
            self.b_loss.append(mean_loss)

        with tf.name_scope("gradient_descent"):
            disc_params = [var for var in tf.trainable_variables() if name_scope in var.name]
            grads, norm = tf.clip_by_global_norm(tf.gradients(mean_loss, disc_params), config.max_grad_norm)
            # optimizer = tf.train.GradientDescentOptimizer(self.lr)
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            train_op = optimizer.apply_gradients(zip(grads, disc_params), global_step=self.global_step)
            self.b_train_op.append(train_op)

        all_variables = [v for v in tf.global_variables() if name_scope in v.name]
        self.saver = tf.train.Saver(all_variables)
