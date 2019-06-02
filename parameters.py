import torch
import os
import codecs

# Device
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Default word tokens
PAD_token = 0  # Used for padding short sentences
EOS_token = 1  # End-of-sentence token
UNK_token = 2  # Unknowen token (out of vocabulary)
SOS_token = 3  # Start-of-sentence token

# Data Hyperparams
BATCH_SIZE = 32
BATCH_SIZE_FAST = 16
BATCH_SIZE_TEST = 128
BATCH_SIZE_VALI = 32
EPOCH = 10
EPOCH_FAST_CONVERGENCE = 20
EPOCH_RNN = 20
SRC_LENGTH = 500
TGT_LENGTH = 70

# Load pretrained embeddings
# Choose embeddings with the same dimension as hidden units
EMBED_PATH_SRC = "./data/pretrained_embedding_512_src.dat"
EMBED_PATH_TGT = "./data/pretrained_embedding_512_tgt.dat"

# training process params
CLIP = 5.0
CONTINUE_TRAIN = 0

# Load test data
TEST_DATA_PATH = [
    "./data/test.txt.src.onehot", "./data/test.txt.tgt.tagged.onehot",
    "./data/test.txt.tgt.tagged.mask", "./data/test.txt.tgt.tagged.gold"
]

# Load train data
TRAIN_DATA_PATH = [
    "./data/train.txt.src.onehot", "./data/train.txt.tgt.tagged.onehot",
    "./data/train.txt.tgt.tagged.mask", "./data/train.txt.tgt.tagged.gold"
]

# Load valid data
VALID_DATA_PATH = [
    "./data/val.txt.src.onehot", "./data/val.txt.tgt.tagged.onehot",
    "./data/val.txt.tgt.tagged.mask", "./data/val.txt.tgt.tagged.gold"
]

# Encoder & Decoder params
EMBED_SIZE = 512
HIDDEN_SIZE = 512
KERNEL_SIZE_ENC = 5
KERNEL_SIZE_DEC = 3
ENC_LAYERS = 20
DEC_LAYERS = 5

# vocab size
VOCAB_SIZE_SRC = 40000
VOCAB_SIZE_TGT = 10000

# dataset size
# 训练集287227
# 测试集11490
# 验证集13368
TRAIN_SIZE = 287227
TEST_SIZE = 11490
VALI_SIZE = 13368

# for print loss
PRINT_EVERY = 50
PRINT_EVERY_SMALL = 10
VALID_EVERY = 100

# for scheduled sampling
TEACHER_FORCING_RATIO_MAX = 1.0
TEACHER_FORCING_RATIO_MIN = 0.3
TEACHER_FORCING_RATIO = 1.0

# dropout
DROPOUT_RATIO = 0.1