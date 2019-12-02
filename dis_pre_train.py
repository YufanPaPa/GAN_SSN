import config
from discriminator import *
import tensorflow as tf
import numpy as np
import os
import time
import random
from six.moves import xrange
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def eval_model(config_disc, query_set_dev, answer_set_dev, gen_set_dev, session, model):
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
                context_positive, context_negative, current_qa_positive, current_qa_negative = get_random_qa(config_disc,
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


def dis_pre_train(config_disc, config_evl):
    config_evl.keep_prob = 1.0

    print("begin training")

    with tf.Session() as session:

        print("prepare_data")
        query_set_train, answer_set_train, gen_set_train, query_set_dev, answer_set_dev, gen_set_dev, = prepare_data(
            config_disc)

        # query_set is of 3 dimension list: 3er dimension is the length of bucket, 2er dimension is the size of data for
        # current bucket id, and 1er dimension is current sentence ids

        # as for answer_set and gen_set, it's the same thing

        train_bucket_sizes = [len(query_set_train[b]) for b in xrange(len(config_disc.buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]
        for set in query_set_train:
            print("set length: ", len(set))

        model = create_model(session, config_disc, name_scope=config_disc.name_model)

        step_time, loss = 0.0, 0.0
        current_step = 0
        step_loss_summary = tf.Summary()

        train_step = config_disc.dis_pre_train_step
        log_write = open(config_disc.log_path, "w")
        previous_dev_loss = 10
        while train_step > 0:
            train_step -= 1
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            start_time = time.time()

            b_query, b_answer, b_gen = query_set_train[bucket_id], answer_set_train[bucket_id], gen_set_train[bucket_id]
            # we need to propose a method to divide b_query into (q,r) pair
            # train_query, train_answer, train_labels = hier_get_batch(config_disc, len(b_query)-1, b_query, b_answer, b_gen)
            # the size of train_query and train_answer is a 2-dimension list, the 2er dimension is of size
            # batch size, and the 1er dimension is the sequence of sen ids. while for train_labels is of
            # dimension 1.

            context_order, context_unorder, current_sen, train_labels = hier_get_qa_batch(
                config_disc, len(b_query) - 1, b_query, b_answer, b_gen)

            # context_order [batch_size,3,seq_length]
            # context_unorder [batch_size,3,seq_length]
            # current_positive [batch_size,3,seq_length]
            # current_negative [batch_size,3,seq_length]

            # train_query = np.transpose(train_query)  # [seq_len, batch_size]
            # train_answer = np.transpose(train_answer)  # [seq_len, batch_size]

            feed_dict = {}
            feed_dict[model.context_order.name] = context_order
            feed_dict[model.context_unorder.name] = context_unorder
            feed_dict[model.current_unorder.name] = current_sen
            # for i in xrange(config_disc.buckets[bucket_id][0]):
            #     feed_dict[model.query[i].name] = train_query[i]
            # for i in xrange(config_disc.buckets[bucket_id][1]):
            #     feed_dict[model.answer[i].name] = train_answer[i]
            feed_dict[model.target.name] = train_labels

            fetches = [model.b_train_op, model.b_logits[0], model.b_loss[0], model.target]
            train_op, logits, step_loss, target = session.run(fetches, feed_dict)

            step_time += (time.time() - start_time) / config_disc.steps_per_checkpoint
            loss += step_loss / config_disc.steps_per_checkpoint
            current_step += 1

            if current_step % config_disc.steps_per_print == 0:

                disc_loss_value = step_loss_summary.value.add()
                disc_loss_value.tag = config_disc.name_loss
                disc_loss_value.simple_value = float(loss)

                print("logits shape: ", np.shape(logits))  # [batch_size,num_class]

                # softmax operation
                logits = np.transpose(softmax(np.transpose(logits)))  # [batch_size,num_class]

                reward = 0.0
                for logit, label in zip(logits, train_labels):
                    reward += logit[1]  # only for true probility
                reward = reward / len(train_labels)
                print("reward: ", reward)
                # for current batch we obtient the average reward,which is the mean of probability

                print("current_step: %d, step_loss: %.4f" % (current_step, step_loss))

                if current_step % (config_disc.steps_per_checkpoint) == 0:
                    disc_ckpt_dir = os.path.abspath(os.path.join(config_disc.checkpoint_dir))
                    if not os.path.exists(disc_ckpt_dir):
                        os.makedirs(disc_ckpt_dir)
                    disc_model_path = os.path.join(disc_ckpt_dir, "disc.model")
                    eval_loss = eval_model(config_disc, query_set_dev, answer_set_dev, gen_set_dev, session, model)
                    log_write.write("global_step " + str(current_step) + "\n")
                    log_write.write("eval_loss " + str(eval_loss) + "\n")
                    log_write.write("\n")
                    log_write.flush()
                    if eval_loss < previous_dev_loss:
                        print("current_step: %d, save model" % (current_step))
                        previous_dev_loss = eval_loss
                        model.saver.save(session, disc_model_path, global_step=model.global_step)

                step_time, loss = 0.0, 0.0
                sys.stdout.flush()


if __name__ == "__main__":
    dis_pre_train(config.disc_config, config.disc_config)
