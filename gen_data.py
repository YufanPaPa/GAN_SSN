import config
import tensorflow as tf
from generator import *
import random
import numpy as np
from six.moves import xrange
import util

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def gen_data(gen_config):
    vocab, rev_vocab, dev_set, train_set = prepare_data(gen_config)
    # vocab:词典的list形式；rev_vocab:词典与其编号的对应；
    # dev_set、train_set格式为：三维list，第三维是不同的区间范围的bucket list，第二维度是每个bucket list里面的对应训练数据，第一维度是
    # 相应的（query_ids,response_ids) pair

    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(gen_config.buckets))]
    train_total_size = float(sum(train_bucket_sizes))
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    with tf.Session() as sess:
        model = create_model(sess, gen_config, forward_only=True, name_scope=gen_config.name_model)

        disc_train_query = open(os.path.join(gen_config.dis_train_path,"query"),"w")
        disc_train_answer = open(os.path.join(gen_config.dis_train_path, "answer"),"w")
        disc_train_gen = open(os.path.join(gen_config.dis_train_path, "gen"),"w")

        disc_dev_query = open(os.path.join(gen_config.dis_dev_path, "query"),"w")
        disc_dev_answer = open(os.path.join(gen_config.dis_dev_path, "answer"),"w")
        disc_dev_gen = open(os.path.join(gen_config.dis_dev_path, "gen"),"w")

        num_step = 0
        while num_step < 1000:
            print("generating num_step: ", num_step)
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])
            # we choose a random bucket_id according to the distribution of probability

            encoder_inputs, decoder_inputs, target_weights\
                , batch_source_encoder, batch_source_decoder = \
                model.get_batch(train_set, bucket_id, gen_config.batch_size)
            # according to the chosen bucket_id, we get a batch data with the query length and response length
            # matching to the corresponding bucket

            # encoder_inputs will be two dimension lists, 2r dimension is of size current_sequence_length,
            # 1r dimension is of size batch_size

            # decoder_inputs will be two dimension lists, 2r dimension is of size current_sequence_length,
            # 1r dimension is of size batch_size

            # target_weights will be two dimension lists, 2r dimension is of size current_sequence_length,
            # 1r dimension is of size batch_size, if current position has a real token, the value will be 1, otherwise will
            # be 0

            _, _, out_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id,
                                          forward_only=True)
            # out_logits will be a one dimension list with the length of decoder steps,
            # each element is [batch_size,target_vocab_size]

            tokens = []
            resps = []
            for seq in out_logits:
                token = []
                for t in seq:
                    token.append(int(np.argmax(t, axis=0)))
                tokens.append(token)
            # tokens will be of two dimension, 2r dimension is of size decoder steps, and 1r dimension is of size batch_size

            tokens_t = []
            for col in range(len(tokens[0])):
                tokens_t.append([tokens[row][col] for row in range(len(tokens))])
            # tokens_t will be of two dimension, 2r dimension is of size batch_size, and 1r dimension is of size decoder steps

            for seq in tokens_t:
                if util.EOS_ID in seq:
                    resps.append(seq[:seq.index(util.EOS_ID)][:gen_config.buckets[bucket_id][1]])
                else:
                    resps.append(seq[:gen_config.buckets[bucket_id][1]])
            # for each sequence in tokens_t, we will post_process each seq by cutting by EOS_ID and max_decode_steps
            if num_step % 100 == 0:
                for query, answer, resp in zip(batch_source_encoder, batch_source_decoder, resps):

                    answer_str = " ".join([str(rev_vocab[an]) for an in answer][:-1])
                    disc_dev_answer.write(answer_str)
                    disc_dev_answer.write("\n")

                    query_str = " ".join([str(rev_vocab[qu]) for qu in query])
                    disc_dev_query.write(query_str)
                    disc_dev_query.write("\n")

                    resp_str = " ".join([tf.compat.as_str(rev_vocab[output]) for output in resp])

                    disc_dev_gen.write(resp_str)
                    disc_dev_gen.write("\n")

                    disc_dev_answer.flush()
                    disc_dev_query.flush()
                    disc_dev_gen.flush()
            else:
                for query, answer, resp in zip(batch_source_encoder, batch_source_decoder, resps):

                    answer_str = " ".join([str(rev_vocab[an]) for an in answer][:-1])
                    disc_train_answer.write(answer_str)
                    disc_train_answer.write("\n")

                    query_str = " ".join([str(rev_vocab[qu]) for qu in query])
                    disc_train_query.write(query_str)
                    disc_train_query.write("\n")

                    resp_str = " ".join([tf.compat.as_str(rev_vocab[output]) for output in resp])

                    disc_train_gen.write(resp_str)
                    disc_train_gen.write("\n")

                    disc_train_answer.flush()
                    disc_train_query.flush()
                    disc_train_gen.flush()

            num_step += 1

        disc_train_gen.close()
        disc_train_query.close()
        disc_train_answer.close()
        disc_dev_gen.close()
        disc_dev_query.close()
        disc_dev_answer.close()
    pass

if __name__ == "__main__":
    gen_data(config.gen_config)