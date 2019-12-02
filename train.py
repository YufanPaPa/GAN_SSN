import os
import tensorflow as tf
import numpy as np
import sys
import time
from six.moves import xrange
import generator as gens
import discriminator as h_disc
import random
import config
import util

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

gen_config = config.gen_config
disc_config = config.disc_config

def eval_generation_model(gen_config, dev_set, sess, model):
    total_loss = []
    for bucket_id in xrange(len(gen_config.buckets)):
        encoder_size, decoder_size = gen_config.buckets[bucket_id]
        dev_data_size = len(dev_set[bucket_id])
        batch_num_per_dev = int(dev_data_size/gen_config.batch_size)
        for batch_id in range(batch_num_per_dev):
            encoder_inputs, decoder_inputs = [], []
            for item_id in range(batch_id*gen_config.batch_size,(batch_id+1)*gen_config.batch_size):
                encoder_input,decoder_input = dev_set[bucket_id][item_id]
                encoder_pad = [util.PAD_ID] * (encoder_size - len(encoder_input))
                encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
                decoder_pad_size = decoder_size - len(decoder_input) - 1
                decoder_inputs.append([util.GO_ID] + decoder_input +
                                      [util.PAD_ID] * decoder_pad_size)
            # Now we create batch-major vectors from the disc_data selected above.
            batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []
            # Batch encoder inputs are just re-indexed encoder_inputs.
            for length_idx in xrange(encoder_size):
                batch_encoder_inputs.append(
                    np.array([encoder_inputs[batch_idx][length_idx]
                              for batch_idx in xrange(gen_config.batch_size)], dtype=np.int32))
            # batch_encoder_inputs will be two dimension lists, 2r dimension is of size current_sequence_length,
            # 1r dimension is of size batch_size
            # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
            for length_idx in xrange(decoder_size):
                batch_decoder_inputs.append(
                    np.array([decoder_inputs[batch_idx][length_idx]
                              for batch_idx in xrange(gen_config.batch_size)], dtype=np.int32))

                # Create target_weights to be 0 for targets that are padding.
                batch_weight = np.ones(gen_config.batch_size, dtype=np.float32)
                for batch_idx in xrange(gen_config.batch_size):
                    # We set weight to 0 if the corresponding target is a PAD symbol.
                    # The corresponding target is decoder_input shifted by 1 forward.
                    if length_idx < decoder_size - 1:
                        target = decoder_inputs[batch_idx][length_idx + 1]
                    if length_idx == decoder_size - 1 or target == util.PAD_ID:
                        batch_weight[batch_idx] = 0.0
                batch_weights.append(batch_weight)
            _, step_loss = model.step(sess, batch_encoder_inputs, batch_decoder_inputs, batch_weights, bucket_id,
                                         forward_only=False, update_gradient=False)
            total_loss.append(step_loss)
    eval_loss = np.mean(total_loss)
    eval_ppl = np.exp(np.mean(total_loss))
    return (eval_loss,eval_ppl)

def eval_dis_model(config_disc, query_set_dev, answer_set_dev, gen_set_dev, session, model):
    total_loss = []
    for bucket_id in xrange(len(config_disc.buckets)):
        dev_data_size = len(query_set_dev[bucket_id])
        batch_num_per_dev = int(dev_data_size / config_disc.batch_size * 2)
        for batch_id in range(batch_num_per_dev):
            context_order_dev = []
            context_unorder_dev = []
            current_sen_dev = []
            train_labels_dev = []
            for item_id in range(int(batch_id * config_disc.batch_size / 2),
                                 int((batch_id + 1) * config_disc.batch_size / 2)):
                query_answer_set, current_query = get_cur_sen_slices(config_disc, query_set_dev[bucket_id][item_id])
                context_positive, context_negative, current_qa_positive, current_qa_negative = get_random_qa_for_dev(config_disc,
                                                                                                             query_answer_set,
                                                                                                             current_query,
                                                                                                answer_set_dev[bucket_id][item_id],
                                                                                                gen_set_dev[bucket_id][item_id])
                context_order_dev.append(context_positive)
                context_unorder_dev.append(context_negative)
                current_sen_dev.append(current_qa_positive)
                train_labels_dev.append(1)

                context_order_dev.append(context_positive)
                context_unorder_dev.append(context_negative)
                current_sen_dev.append(current_qa_negative)
                train_labels_dev.append(0)

            feed_dict = {}
            feed_dict[model.context_order.name] = context_order_dev
            feed_dict[model.context_unorder.name] = context_unorder_dev
            feed_dict[model.current_unorder.name] = current_sen_dev
            feed_dict[model.target.name] = train_labels_dev

            fetches = [model.b_logits, model.b_loss[0], model.target]
            logits, step_loss, target = session.run(fetches, feed_dict)

            total_loss.append(step_loss)
    return np.mean(total_loss)

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

def get_random_qa(config, query_slice_set, current_query, answer):
    # query_answer_set is a list of q-a sequence
    # we need to get context_qa_order[3,seq_length],context_qa_disorder[3,seq_length],current_qa_disorder[3,seq_length]
    order_sequence = []
    disorder_sequence = []
    current_sequence_positive = []
    current_qa_positive = current_query + [6] + answer
    current_qa_positive = current_qa_positive[:2 * config.max_len] + [util.PAD_ID] * (
        2 * config.max_len - len(current_qa_positive) if 2 * config.max_len > len(current_qa_positive) else 0)
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

        return order_sequence, disorder_sequence, current_sequence_positive

def get_random_qa_for_dev(config, query_slice_set, current_query, answer, generation):
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

# prepare disc_data for discriminator and generator
def disc_train_data(sess, gen_model, vocab, source_inputs, source_outputs,
                    encoder_inputs, decoder_inputs, target_weights, bucket_id, mc_search=False):
    # source_inputs and source_outputs are 2 dimension list, 2r dimension is of size batch_size, and 1r dimension
    # is a list of sentence ids.(not padding)
    train_query, train_answer = [], []
    context_order = []
    context_unorder = []
    current_sen = []
    train_labels = []
    query_len = gen_config.buckets[bucket_id][0]
    answer_len = gen_config.buckets[bucket_id][1]

    for query, answer in zip(source_inputs, source_outputs):
        query_answer_set, current_query = get_cur_sen_slices(disc_config, query)
        answer = answer[:-1]
        order_sequence, disorder_sequence, current_sequence_positive = get_random_qa(disc_config,query_answer_set,current_query,
                                                                                    answer)
        context_order.append(order_sequence)
        context_unorder.append(disorder_sequence)
        current_sen.append(current_sequence_positive)
        train_labels.append(1)



    def decoder(num_roll):
        for _ in xrange(num_roll):
            _, _, output_logits = gen_model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id,
                                                 forward_only=True, mc_search=mc_search)
            # out_logits will be a one dimension list with the length of decoder steps,
            # each element is [batch_size,target_vocab_size]

            seq_tokens = []
            resps = []
            for seq in output_logits:
                row_token = []
                for t in seq:
                    row_token.append(int(np.argmax(t, axis=0)))
                seq_tokens.append(row_token)

            # seq_tokens will be of two dimension, 2r dimension is of size decoder steps, and 1r dimension is of size batch_size
            seq_tokens_t = []
            for col in range(len(seq_tokens[0])):
                seq_tokens_t.append([seq_tokens[row][col] for row in range(len(seq_tokens))])

            # seq_tokens_t will be of 2 dimension, 2r dimension is of size batch_size, and 1r dimension is of size sequence length

            for seq in seq_tokens_t:
                if util.EOS_ID in seq:
                    resps.append(seq[:seq.index(util.EOS_ID)][:gen_config.buckets[bucket_id][1]])
                else:
                    resps.append(seq[:gen_config.buckets[bucket_id][1]])
            # we cut each sequence in seq_tokens_t by the EOS tag and the max length defined by bucket.
            for (query, generation) in zip(source_inputs, resps):
                query_answer_set, current_query = get_cur_sen_slices(disc_config, query)
                order_sequence, disorder_sequence, current_sequence_negative = get_random_qa(disc_config,
                                                                                             query_answer_set,
                                                                                             current_query,
                                                                                             generation)
                context_order.append(order_sequence)
                context_unorder.append(disorder_sequence)
                current_sen.append(current_sequence_negative)
                train_labels.append(0)

            # we generate the answer using generation model and we padding the generated result

        return context_order, context_unorder, current_sen, train_labels

    if mc_search:
        context_order, context_unorder, current_sen, train_labels = decoder(gen_config.beam_size)
    else:
        context_order, context_unorder, current_sen, train_labels = decoder(1)

    return context_order, context_unorder, current_sen, train_labels


def softmax(x):
    prob = np.exp(x) / np.sum(np.exp(x), axis=0)
    return prob


# discriminator api
def disc_step(sess, bucket_id, disc_model, context_order, context_unorder, current_sen, train_labels, forward_only=False):
    # train_query,train_answer:[sequence_length, batch_size*2]
    feed_dict = {}
    feed_dict[disc_model.context_order.name] = context_order
    feed_dict[disc_model.context_unorder.name] = context_unorder
    feed_dict[disc_model.current_unorder.name] = current_sen
    feed_dict[disc_model.target.name] = train_labels

    loss = 0.0
    if forward_only:
        fetches = [disc_model.b_logits[0]]
        logits = sess.run(fetches, feed_dict)
        logits = logits[0]
    else:
        fetches = [disc_model.b_train_op, disc_model.b_loss[0], disc_model.b_logits[0]]
        train_op, loss, logits = sess.run(fetches,feed_dict)
    # softmax operation
    logits = np.transpose(softmax(np.transpose(logits)))

    reward, gen_num = 0.0, 0
    for logit, label in zip(logits, train_labels):
        if int(label) == 0:
            reward += logit[1]
            gen_num += 1
    reward = reward / gen_num

    return reward, loss


# Adversarial Learning for Neural Dialogue Generation
def al_train():
    with tf.Session() as sess:

        vocab, rev_vocab, dev_set, train_set = gens.prepare_data(gen_config)
        query_set_train, answer_set_train, gen_set_train, query_set_dev, answer_set_dev, gen_set_dev, = h_disc.prepare_data(disc_config)
        # vocab:a dict, word to id mapping；rev_vocab: a list, id to word mapping；
        # dev_set、train_set格式为：four dimension list，4er dimension is of length bucket size, 3er dimension is of length
        # size of current bucket, 2er dimension is a pair of query ids and response ids, and 1er dimension is a list of
        # sequence ids.
        for set in train_set:
            print("al train len: ", len(set))

        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(gen_config.buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

        disc_model = h_disc.create_model(sess, disc_config, disc_config.name_model)
        gen_model = gens.create_model(sess, gen_config, forward_only=False, name_scope=gen_config.name_model)

        current_step = 0
        step_time, disc_loss, gen_loss, t_loss, batch_reward = 0.0, 0.0, 0.0, 0.0, 0.0
        gen_loss_summary = tf.Summary()
        disc_loss_summary = tf.Summary()
        log_write = open(gen_config.log_path_ensemble, "w")

        while True:
            current_step += 1
            start_time = time.time()
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                         if train_buckets_scale[i] > random_number_01])

            print("==================Update Discriminator: %d=====================" % current_step)
            # 1.Sample (X,Y) from real disc_data
            encoder_inputs, decoder_inputs, target_weights, source_inputs, source_outputs = gen_model.get_batch(train_set, bucket_id, gen_config.batch_size)
            # encoder_inputs will be two dimension lists, 2r dimension is of size current_sequence_length,
            # 1r dimension is of size batch_size(already padding)

            # decoder_inputs will be two dimension lists, 2r dimension is of size current_sequence_length,
            # 1r dimension is of size batch_size(already padding)

            # target_weights will be two dimension lists, 2r dimension is of size current_sequence_length,
            # 1r dimension is of size batch_size, if current position has a real token, the value will be 1, otherwise will
            # be 0(already padding)

            # source_inputs and source_outputs are 2 dimension list, 2r dimension is of size batch_size, and 1r dimension
            # is a list of sentence ids.(not padding)

            # 2.Sample (X,Y) and (X, ^Y) through ^Y ~ G(*|X)
            context_order, context_unorder, current_sen, train_labels = disc_train_data(sess, gen_model, vocab, source_inputs, source_outputs,
                                                        encoder_inputs, decoder_inputs, target_weights, bucket_id, mc_search=False)
            # context_order, context_unorder, current_sen:[batch_size,3,seq_length]
            # train_labels:[batch_size]

            # if current_step % 200 == 0:
            #     print("train_query: ", len(train_query))
            #     print("train_answer: ", len(train_answer))
            #     print("train_labels: ", len(train_labels))
            #     for i in xrange(len(train_query)):
            #         print("label: ", train_labels[i])
            #         print("train_answer_sentence: ", train_answer[i])
            #         print(" ".join([tf.compat.as_str(rev_vocab[output]) for output in train_answer[i]]))


            # 3.Update D using (X, Y ) as positive examples and(X, ^Y) as negative examples
            _, disc_step_loss = disc_step(sess, bucket_id, disc_model, context_order, context_unorder, current_sen, train_labels, forward_only=False)
            disc_loss += disc_step_loss / disc_config.steps_per_checkpoint

            print("==================Update Generator: %d=========================" % current_step)
            # 1.Sample (X,Y) from real disc_data
            update_gen_data = gen_model.get_batch(train_set, bucket_id, gen_config.batch_size)
            encoder, decoder, weights, source_inputs, source_outputs = update_gen_data
            # encoder will be two dimension lists, 2r dimension is of size current_sequence_length,
            # 1r dimension is of size batch_size(already padding)

            # decoder will be two dimension lists, 2r dimension is of size current_sequence_length,
            # 1r dimension is of size batch_size(already padding)

            # weights will be two dimension lists, 2r dimension is of size current_sequence_length,
            # 1r dimension is of size batch_size, if current position has a real token, the value will be 1, otherwise will
            # be 0(already padding)

            # source_inputs and source_outputs are 2 dimension list, 2r dimension is of size batch_size, and 1r dimension
            # is a list of sentence ids.(not padding)

            # 2.Sample (X,Y) and (X, ^Y) through ^Y ~ G(*|X) with Monte Carlo search
            context_order, context_unorder, current_sen, train_labels = disc_train_data(sess, gen_model, vocab, source_inputs, source_outputs,
                                                                encoder, decoder, weights, bucket_id, mc_search=True)


            # train_query and train_answer will be two dimension lists, 2r dimension is of size batch_size*?(depende on
            # the MC setting, 1r dimension is of size sequence length.
            # train_labels will be 1 dimension of size batch_size*?

            # if current_step % 200 == 0:
            #     for i in xrange(len(train_query)):
            #         print("label: ", train_labels[i])
            #         print(" ".join([tf.compat.as_str(rev_vocab[output]) for output in train_answer[i]]))

            # train_query,train_answer:[sequence_length, batch_size*?]

            # 3.Compute Reward r for (X, ^Y ) using D.---based on Monte Carlo search
            reward, _ = disc_step(sess, bucket_id, disc_model, context_order, context_unorder, current_sen, train_labels, forward_only=True)
            batch_reward += reward / gen_config.steps_per_checkpoint
            print("step_reward: ", reward)

            # 4.Update G on (X, ^Y ) using reward r
            gan_adjusted_loss, gen_step_loss, _ =gen_model.step(sess, encoder, decoder, weights, bucket_id, forward_only=False,
                                           reward=reward, up_reward=True, debug=True)

            # as for current selected batch:encoder,we use current generation model to generate multi response and based
            # on these (query,response) pairs, the disc model will give a reward, and this reward will modifier the current
            # generation loss for current batch
            gen_loss += gen_step_loss / gen_config.steps_per_checkpoint

            print("gen_step_loss: ", gen_step_loss)
            print("gen_step_adjusted_loss: ", gan_adjusted_loss)

            # 5.Teacher-Forcing: Update G on (X, Y )
            t_adjusted_loss, t_step_loss, a = gen_model.step(sess, encoder, decoder, weights, bucket_id, forward_only=False)
            t_loss += t_step_loss / gen_config.steps_per_checkpoint
           
            print("t_step_loss: ", t_step_loss)
            print("t_adjusted_loss", t_adjusted_loss)

            if current_step % gen_config.steps_per_print == 0:

                step_time += (time.time() - start_time) / gen_config.steps_per_checkpoint

                print("current_steps: %d, step time: %.4f, disc_loss: %.3f, gen_loss: %.3f, t_loss: %.3f, reward: %.3f"
                      %(current_step, step_time, disc_loss, gen_loss, t_loss, batch_reward))

                disc_loss_value = disc_loss_summary.value.add()
                disc_loss_value.tag = disc_config.name_loss
                disc_loss_value.simple_value = float(disc_loss)

                gen_global_steps = sess.run(gen_model.global_step)
                gen_loss_value = gen_loss_summary.value.add()
                gen_loss_value.tag = gen_config.name_loss
                gen_loss_value.simple_value = float(gen_loss)
                t_loss_value = gen_loss_summary.value.add()
                t_loss_value.tag = gen_config.teacher_loss
                t_loss_value.simple_value = float(t_loss)
                batch_reward_value = gen_loss_summary.value.add()
                batch_reward_value.tag = gen_config.reward_name
                batch_reward_value.simple_value = float(batch_reward)

                if current_step % (gen_config.steps_per_checkpoint) == 0:
                    print("current_steps: %d, save disc model" % current_step)
                    disc_ckpt_dir = disc_config.checkpoint_dir_ensemble
                    if not os.path.exists(disc_ckpt_dir):
                        os.makedirs(disc_ckpt_dir)
                    disc_model_path = os.path.join(disc_ckpt_dir, "disc.model")
                    eval_loss_disc = eval_dis_model(disc_config, query_set_dev, answer_set_dev, gen_set_dev, sess, disc_model)
                    disc_model.saver.save(sess, disc_model_path, global_step=disc_model.global_step)

                    print("current_steps: %d, save gen model" % current_step)
                    gen_ckpt_dir = gen_config.checkpoint_dir_ensemble
                    if not os.path.exists(gen_ckpt_dir):
                        os.makedirs(gen_ckpt_dir)
                    gen_model_path = os.path.join(gen_ckpt_dir, "gen.model")
                    (eval_loss_generation, eval_ppl_generation) = eval_generation_model(gen_config, dev_set, sess, gen_model)
                    gen_model.saver.save(sess, gen_model_path, global_step=gen_model.global_step)

                    log_write.write("global_step " + str(gen_global_steps) + "\n")
                    log_write.write("eval_loss_generation " + str(eval_loss_generation) + "\n")
                    log_write.write("eval_ppl_generation " + str(eval_ppl_generation) + "\n")
                    log_write.write("eval_loss_disc " + str(eval_loss_disc) + "\n")
                    log_write.write("\n")
                    log_write.flush()

                step_time, disc_loss, gen_loss, t_loss, batch_reward = 0.0, 0.0, 0.0, 0.0, 0.0
                sys.stdout.flush()


def main(_):
    al_train()

if __name__ == "__main__":
    tf.app.run()
