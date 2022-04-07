import os
from configuration import *
from data_eda import data_trans,word2emb_get
import numpy as np
from dataloader import data_iter
from train import model_lv1,model_lv3, epoch_train, optimizer_lv1, optimizer_lv3
import random
random.seed(2021) 

if __name__ == '__main__':
    if not os.path.exists(write_data_path[0]):
        # 生成数据
        data_trans(origin_train_path, write_data_path)
        # 词向量生成
        word2emb_get(write_data_path, emb_dim)

    # topic lv1 train
    train_dl_lv1, val_dl_lv1 = data_iter(write_data_path ,1)
    epoch_train(epoch_num, model_lv1, optimizer_lv1,train_dl_lv1, val_dl_lv1,best_model_out_path_lv1)

    # topic lv3 train
    train_dl_lv3, val_dl_lv3 = data_iter(write_data_path ,3)
    epoch_train(epoch_num, model_lv3, optimizer_lv3,train_dl_lv3, val_dl_lv3,best_model_out_path_lv3)
