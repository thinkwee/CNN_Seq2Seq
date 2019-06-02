from visdom import Visdom
import _pickle as pickle
import torch
import time

# def tensor_to_numpy(tensor_list):
#     loss = tensor_list[0].view(1)
#     for i in range(0, len(tensor_list)):
#         loss = torch.cat([loss, tensor_list[i].view(1)])
#     return loss.cpu().numpy()

# train_loss = pickle.load(open("../model_check/train_loss.dat", "rb"))
# train_loss = tensor_to_numpy(train_loss)
# print(train_loss)
# viz = Visdom()
# assert viz.check_connection()

# viz.line(Y = train_loss)

viz = Visdom(env=u'main')
viz.text(time.strftime("STARTS AT %a %b %d %H:%M:%S %Y \n", time.localtime()), win='summary')

