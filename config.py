import os

# configuration options for discriminator network
class disc_config(object):
    batch_size = 64
    lr = 0.2
    lr_decay = 0.9
    vocab_size = 15005
    embed_dim = 512
    steps_per_print = 10
    steps_per_checkpoint = 20
    num_layers = 1
    train_dir = './train_data/data_ubuntu/data_discri/train'
    dev_dir = './train_data/data_ubuntu/data_discri/dev'
    vocab_dir = "./train_data/data_ubuntu/vocab/word_dict_encode"
    checkpoint_dir = "./train_data/data_ubuntu/checkpoints_disc"
    checkpoint_dir_ensemble = "./train_data/data_ubuntu/checkpoints_disc_ensemble"
    log_path = "./train_data/data_ubuntu/log_dis"
    name_model = "disc_model"
    name_loss = "disc_loss"
    max_len = 50
    piece_size = batch_size * steps_per_checkpoint
    piece_dir = "./dis_data/batch_piece/"
    valid_num = 100
    init_scale = 0.1
    num_class = 2
    keep_prob = 0.5
    max_grad_norm = 5
    buckets = [(180, 12), (360, 24), (540, 36), (720, 48)]
    dis_pre_train_step = 80000


# configuration options for generator network
class gen_config(object):
    beam_size = 7
    learning_rate = 0.0001
    learning_rate_decay_factor = 0.99
    max_gradient_norm = 5.0
    batch_size = 64
    emb_dim = 256
    num_layers = 1
    vocab_size = 15005
    train_dir = "./train_data/data_ubuntu/data_generation/train"
    dev_dir = "./train_data/data_ubuntu/data_generation/dev"
    test_dir = "./train_data/data_ubuntu/data_generation/test"
    test_result = "./train_data/data_ubuntu/test.result"
    vocab_dir = "./train_data/data_ubuntu/vocab/word_dict_encode"
    checkpoint_dir = "./train_data/data_ubuntu/checkpoints_generation"
    checkpoint_dir_ensemble = "./train_data/data_ubuntu/checkpoints_generation_ensemble"
    log_path = "./train_data/data_ubuntu/log_generation"
    log_path_ensemble = "./train_data/data_ubuntu/log_ensemble"
    dis_train_path = './train_data/data_ubuntu/data_discri/train'
    dis_dev_path = './train_data/data_ubuntu/data_discri/dev'
    name_model = "st_model"
    name_loss = "gen_loss"
    teacher_loss = "teacher_loss"
    reward_name = "reward"
    max_train_data_size = 0
    steps_per_checkpoint = 20
    steps_per_print = 10
    buckets = [(180, 12), (360, 24), (540, 36), (720, 48)]
    buckets_concat = [(5, 10), (10, 15), (20, 25), (40, 50), (100, 50)]
    max_query_len = 700
    max_response_len = 50
    gen_pre_train_step = 400000