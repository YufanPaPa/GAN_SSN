import config
import os
import sys
import numpy as np
from tensorflow.python.platform import gfile
import tensorflow as tf
from generator import *
import re

_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
_WORD_SPLIT = re.compile(r"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def basic_tokenizer(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        if type(space_separated_fragment) == bytes:
            space_separated_fragment = space_separated_fragment.decode()
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w for w in words if w]

def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    return [vocabulary.get(_DIGIT_RE.sub("0", w), UNK_ID) for w in words]

def data_to_token_ids(data_path, query_path, answer_path, vocabulary,
                      tokenizer=None, normalize_digits=True):
    if not gfile.Exists(answer_path):
        print("Tokenizing disc_data in %s" % data_path)
        with gfile.GFile(data_path, mode="r") as data_file:
            with gfile.GFile(query_path, mode="w") as query_file, gfile.GFile(answer_path, mode="w") as answer_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  tokenizing line %d" % counter)
                    infos = line.strip().split("\t")
                    query = " [SEP] ".join(infos[:-2]).strip()
                    answer = infos[-1].strip()
                    query_ids = sentence_to_token_ids(query, vocabulary, tokenizer,
                                                      normalize_digits)
                    answer_ids = sentence_to_token_ids(answer, vocabulary, tokenizer,
                                                       normalize_digits)
                    query_file.write(" ".join([str(tok) for tok in query_ids]) + "\n")
                    answer_file.write(" ".join([str(tok) for tok in answer_ids]) + "\n")

def read_data(config, source_path, target_path, max_size=None):
    data_set = [[] for _ in config.buckets]
    with gfile.GFile(source_path, mode="r") as source_file:
        with gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 100000 == 0:
                    print("  reading disc_data line %d" % counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()][:config.max_query_len]
                target_ids = [int(x) for x in target.split()][:config.max_response_len]
                target_ids.append(util.EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(config.buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set

def eval_model(gen_config, dev_set, sess, model, rev_vocab):
    total_loss = []
    result_file = open(gen_config.test_result,"w")
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
            _, _, output_logits = model.step(sess, batch_encoder_inputs, batch_decoder_inputs, batch_weights, bucket_id,
                                                 forward_only=True, mc_search=False)

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
            batch_decoder_inputs = np.transpose(batch_decoder_inputs)
            for (ground_truth,generation) in zip(batch_decoder_inputs, resps):
                ground_truth = " ".join([rev_vocab[int(id)] for id in ground_truth if int(id)!=1 and int(id)!=2 and int(id)!=0])
                generation = " ".join([rev_vocab[int(id)] for id in generation if int(id)!=1 and int(id)!=2])
                result_file.write(ground_truth.strip()+"\t"+generation.strip()+"\n")

            # we cut each sequence in seq_tokens_t by the EOS tag and the max length defined by bucket.
            total_loss.append(step_loss)
    result_file.close()
    eval_loss = np.mean(total_loss)
    eval_ppl = np.exp(np.mean(total_loss))
    return (eval_loss,eval_ppl)

def test(gen_config):
    with tf.Session() as sess:
        model = create_model(sess, gen_config, forward_only=True, name_scope=gen_config.name_model)

        # sum = 0
        # for tv in tf.trainable_variables():
        #     shape = list(tv.shape)
        #     cur_sum = 1
        #     for i in shape:
        #         cur_sum = cur_sum * int(i)
        #     sum += cur_sum
        # print(sum)
        # assert 0==1

        vocab_path = gen_config.vocab_dir
        vocab, rev_vocab = util.initialize_vocabulary(vocab_path)
        test_path = gen_config.test_dir

        answer_test_ids_path = test_path + (".ids%d.answer" % gen_config.vocab_size)
        query_test_ids_path = test_path + (".ids%d.query" % gen_config.vocab_size)
        data_to_token_ids(test_path, query_test_ids_path, answer_test_ids_path, vocab, tokenizer=None)
        test_set = read_data(gen_config, query_test_ids_path, answer_test_ids_path, gen_config.max_train_data_size)
        (eval_loss, eval_ppl) = eval_model(gen_config, test_set, sess, model,rev_vocab)
        print(eval_ppl)


if __name__ == '__main__':
    test(config.gen_config)
