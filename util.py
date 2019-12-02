import os
import re
from six.moves import urllib
from tensorflow.python.platform import gfile
import tensorflow as tf

# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(r"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")


def basic_tokenizer(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        if type(space_separated_fragment) == bytes:
            space_separated_fragment = space_separated_fragment.decode()
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path_list, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from disc_data %s" % (vocabulary_path, data_path_list))
        vocab = {}
        for data_path in data_path_list:
            with gfile.GFile(data_path, mode="r") as f:
                counter = 0
                for line in f:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  processing line %d" % counter)
                    line = tf.compat.as_str_any(line)
                    tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                    for w in tokens:
                        word = _DIGIT_RE.sub("0", w) if normalize_digits else w
                        if word in vocab:
                            vocab[word] += 1
                        else:
                            vocab[word] = 1

        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]
        with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + "\n")


def initialize_vocabulary(vocabulary_path):
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip().split("\t")[0] for line in rev_vocab]
        rev_vocab = _START_VOCAB + rev_vocab
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


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
                    query = " [SEP] ".join(infos[:-1]).strip()
                    answer = infos[-1].strip()
                    query_ids = sentence_to_token_ids(query, vocabulary, tokenizer,
                                                      normalize_digits)
                    answer_ids = sentence_to_token_ids(answer, vocabulary, tokenizer,
                                                       normalize_digits)
                    query_file.write(" ".join([str(tok) for tok in query_ids]) + "\n")
                    answer_file.write(" ".join([str(tok) for tok in answer_ids]) + "\n")

def data_to_token_ids_single(train_path, save_path, vocabulary, tokenizer=None, normalize_digits=True):
    if not gfile.Exists(save_path):
        print("Tokenizing disc_data in %s" % save_path)
        with gfile.GFile(train_path, mode="r") as data_file:
            with gfile.GFile(save_path, mode="w") as save_file:
                counter = 0
                for line in data_file:
                    info = line.strip()
                    counter += 1
                    if counter % 100000 == 0:
                        print("  tokenizing line %d" % counter)
                    sen_ids = sentence_to_token_ids(info, vocabulary, tokenizer,
                                                      normalize_digits)
                    save_file.write(" ".join([str(tok) for tok in sen_ids]) + "\n")



def prepare_chitchat_data(train_path, dev_path, vocabulary, vocabulary_size, tokenizer=None):
    answer_train_ids_path = train_path + (".ids%d.answer" % vocabulary_size)
    query_train_ids_path = train_path + (".ids%d.query" % vocabulary_size)
    data_to_token_ids(train_path, query_train_ids_path, answer_train_ids_path, vocabulary, tokenizer)

    # Create token ids for the development disc_data.
    answer_dev_ids_path = dev_path + (".ids%d.answer" % vocabulary_size)
    query_dev_ids_path = dev_path + (".ids%d.query" % vocabulary_size)
    data_to_token_ids(dev_path, query_dev_ids_path, answer_dev_ids_path, vocabulary, tokenizer)

    return (query_train_ids_path, answer_train_ids_path,
            query_dev_ids_path, answer_dev_ids_path)


def hier_prepare_disc_data(train_dir, dev_dir, vocabulary, vocabulary_size, tokenizer=None):

    train_query_path = os.path.join(train_dir, "query")
    train_answer_path = os.path.join(train_dir, "answer")
    train_gen_path = os.path.join(train_dir, "gen")

    dev_query_path = os.path.join(dev_dir, "query")
    dev_answer_path = os.path.join(dev_dir, "answer")
    dev_gen_path = os.path.join(dev_dir, "gen")

    # Create token ids for the training disc_data.
    query_train_ids_path = train_query_path + (".ids%d.query" % vocabulary_size)
    answer_train_ids_path = train_answer_path + (".ids%d.answer" % vocabulary_size)
    gen_train_ids_path = train_gen_path + (".ids%d.gen" % vocabulary_size)

    data_to_token_ids_single(train_query_path, query_train_ids_path, vocabulary, tokenizer)
    data_to_token_ids_single(train_answer_path, answer_train_ids_path, vocabulary, tokenizer)
    data_to_token_ids_single(train_gen_path, gen_train_ids_path, vocabulary, tokenizer)

    # Create token ids for the training disc_data.
    query_dev_ids_path = dev_query_path + (".ids%d.query" % vocabulary_size)
    answer_dev_ids_path = dev_answer_path + (".ids%d.answer" % vocabulary_size)
    gen_dev_ids_path = dev_gen_path + (".ids%d.gen" % vocabulary_size)

    data_to_token_ids_single(dev_query_path, query_dev_ids_path, vocabulary, tokenizer)
    data_to_token_ids_single(dev_answer_path, answer_dev_ids_path, vocabulary, tokenizer)
    data_to_token_ids_single(dev_gen_path, gen_dev_ids_path, vocabulary, tokenizer)


    return (query_train_ids_path, answer_train_ids_path, gen_train_ids_path,
            query_dev_ids_path, answer_dev_ids_path, gen_dev_ids_path)
