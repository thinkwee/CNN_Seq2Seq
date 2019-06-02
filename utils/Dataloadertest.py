from parameters import *
import torch
import _pickle as pickle
import torch.utils.data as Data

# 加载训练数据做训练
train_src_onehot = pickle.load(open(TRAIN_DATA_PATH[0], "rb"))
train_tgt_onehot = pickle.load(open(TRAIN_DATA_PATH[1], "rb"))
train_len_mask = pickle.load(open(TRAIN_DATA_PATH[2], "rb"))
train_output_onehot = pickle.load(open(TRAIN_DATA_PATH[3], "rb"))

# 加载验证数据
valid_src_onehot = pickle.load(open(VALID_DATA_PATH[0], "rb"))
valid_tgt_onehot = pickle.load(open(VALID_DATA_PATH[1], "rb"))
valid_len_mask = pickle.load(open(VALID_DATA_PATH[2], "rb"))
valid_output_onehot = pickle.load(open(VALID_DATA_PATH[3], "rb"))

# 创建data loader
train_dataset = Data.TensorDataset(train_src_onehot, train_tgt_onehot, train_len_mask,
                                   train_output_onehot)
train_loader = Data.DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)



# for step, (train_encoder_input, train_decoder_input, train_decoder_len_mask,
#                train_decoder_gold) in enumerate(train_loader):
#                print(train_decoder_len_mask)
#                break

# s = input("pause")
loader_iter = train_loader.__iter__()
train_encoder_input, train_decoder_input, train_decoder_len_mask,train_decoder_gold = next(loader_iter)
print(train_decoder_len_mask)
s = input("pause")

for step, (vali_encoder_input, vali_decoder_input, vali_decoder_len_mask,
               vali_decoder_gold) in enumerate(valid_loader):
               if step == 100:
                   break
s = input("pause")
