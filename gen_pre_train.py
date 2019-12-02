import config
import os
import tensorflow as tf
from generator import *
import numpy as np
import time
import sys
import math
import random
from six.moves import xrange

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def eval_model(gen_config, dev_set, sess, model):
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
            step_loss = model.step(sess, batch_encoder_inputs, batch_decoder_inputs, batch_weights, bucket_id,
                                         forward_only=False, update_gradient=False)
            total_loss.append(step_loss)
    eval_loss = np.mean(total_loss)
    eval_ppl = np.exp(np.mean(total_loss))
    return (eval_loss,eval_ppl)


def gen_pre_train(gen_config):
    vocab, rev_vocab, dev_set, train_set = prepare_data(gen_config)
    # vocab:词典的list形式；rev_vocab:词典与其编号的对应；
    # dev_set、train_set格式为：三维list，第三维是不同的区间范围的bucket list，第二维度是每个bucket list里面的对应训练数据，第一维度是
                              #相应的（query_ids,response_ids) pair
    for b_set in train_set:
        print("b_set: ", len(b_set))

    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True
    log_write = open(gen_config.log_path,"w")
    with tf.Session(config=session_config) as sess:
        # Create model.

        print("Creating %d layers of %d units." % (gen_config.num_layers, gen_config.emb_dim))
        model = create_model(sess, gen_config, forward_only=False, name_scope=gen_config.name_model)
        #check if any ckpt exists,if exists, we will restore this model,else,we will initial this model.

        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(gen_config.buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]


        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_dev_ppl = 10000

        gen_loss_summary = tf.Summary()

        train_step = gen_config.gen_pre_train_step
        while train_step>0:
            train_step -= 1
            # Choose a bucket according to disc_data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])
            # as for our train_data suppose the train_buckets_scale is [0.15,0.2,0.35,0.4], then the prob will be [0.15,
            # 0.05,0.15,0.65]

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights, batch_source_encoder, batch_source_decoder = model.get_batch(
                train_set, bucket_id, gen_config.batch_size)

            # encoder_inputs will be two dimension lists, 2r dimension is of size current_sequence_length,
            # 1r dimension is of size batch_size

            # decoder_inputs will be two dimension lists, 2r dimension is of size current_sequence_length,
            # 1r dimension is of size batch_size

            # target_weights will be two dimension lists, 2r dimension is of size current_sequence_length,
            # 1r dimension is of size batch_size, if current position has a real token, the value will be 1, otherwise will
            # be 0


            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only=False)

            step_time += (time.time() - start_time) / gen_config.steps_per_checkpoint
            loss += step_loss / gen_config.steps_per_print
            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % gen_config.steps_per_print == 0:

                bucket_value = gen_loss_summary.value.add()
                bucket_value.tag = gen_config.name_loss
                bucket_value.simple_value = float(loss)

                # Print statistics for the previous epoch.
                perplexity = math.exp(loss) #if loss < 300 else float('inf')
                print ("global step %d learning rate %.4f step-time %.2f perplexity "
                       "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                 step_time, perplexity))

                # Save checkpoint and zero timer and loss.
                if current_step % (gen_config.steps_per_checkpoint) == 0:
                    gen_ckpt_dir = gen_config.checkpoint_dir
                    if not os.path.exists(gen_ckpt_dir):
                        os.makedirs(gen_ckpt_dir)
                    checkpoint_path = os.path.join(gen_ckpt_dir, "chitchat.model")
                    (eval_loss, eval_ppl) = eval_model(gen_config, dev_set, sess, model)
                    log_write.write("global_step "+str(current_step)+"\n")
                    log_write.write("eval_loss "+str(eval_loss)+"\n")
                    log_write.write("eval_ppl " + str(eval_ppl)+"\n")
                    log_write.write("\n")
                    log_write.flush()
                    if eval_ppl < previous_dev_ppl:
                        print("current_step: %d, save model" % (current_step))
                        previous_dev_ppl = eval_ppl
                        model.saver.save(sess, checkpoint_path, global_step=model.global_step)

                step_time, loss = 0.0, 0.0
                sys.stdout.flush()

if __name__ == "__main__":
    gen_pre_train(config.gen_config)