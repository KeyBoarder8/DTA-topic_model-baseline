import numpy as np
import random
import pickle as pkl
from tqdm import tqdm
from configuration import *
import jieba_fast as jieba
import os
random.seed(2021)

def label_flat_mapping(label_list, label_mapping):
    label_result = []
    label_list[0]['start_idx'] = 0
    for i in label_list:
        item = i['topic_role_type'].strip()
        if 'B_'+item not in label_mapping:
            item = '其他'
        label_result += [label_mapping['B_'+item]]+[label_mapping['I_'+item]] * (i['end_idx']-i['start_idx'])
    return label_result

def data_read(data_path):
    text_all, label_lv1_all, label_lv3_all = [],[],[]
    with open(data_path,'r') as f:
        for line in f:
            dialog_info = eval(line)
            text_info = [i['text'] for i in dialog_info['dialogues']]
            label_info_lv1 = [i for i in dialog_info['topics'] if i['topic_level']==1]
            label_info_lv3 = [i for i in dialog_info['topics'] if i['topic_level']==2]
            label_info_lv1 = label_flat_mapping(label_info_lv1,label_mapping_lv1)
            label_info_lv3 = label_flat_mapping(label_info_lv3,label_mapping_lv3)
            text_all.append(text_info)
            label_lv1_all.append(label_info_lv1)
            label_lv3_all.append(label_info_lv3)
    return text_all, label_lv1_all, label_lv3_all

def train_file_gen(text_list, label_list_lv1, label_list_lv3, write_file_path):
    write_lines = []
    for i in range(len(text_list)):
        for j in range(len(text_list[i])):
            if len(text_list[i][j].strip()) != 0 and '\n' not in text_list[i][j]:
                write_lines.append(' '.join(list(jieba.cut(str(text_list[i][j])))) + '\t' +str(label_list_lv1[i][j]) + '\t'  \
                                +str(label_list_lv3[i][j]) + '\n')
        write_lines[-1] = write_lines[-1] + '\n'
    with open(write_file_path, 'w+') as f:
        f.writelines(write_lines)


def data_trans(origin_file_path, write_data_path):
    text_all, label_lv1_all, label_lv3_all = data_read(origin_file_path)
    train_idx = int(len(text_all)*0.9)
    text_train, label_lv1_train, label_lv3_train = text_all[:train_idx], label_lv1_all[:train_idx], label_lv3_all[:train_idx]
    text_val, label_lv1_val, label_lv3_val = text_all[train_idx:], label_lv1_all[train_idx:], label_lv3_all[train_idx:]
    train_file_gen(text_all, label_lv1_all, label_lv3_all, write_data_path[0])
    train_file_gen(text_train, label_lv1_train, label_lv3_train, write_data_path[1])
    train_file_gen(text_val, label_lv1_val, label_lv3_val, write_data_path[2])
    print('数据生成完毕.......')

def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic

def word2emb_get(write_data_path,emb_dim):
    '''提取预训练词向量'''
    tokenizer = lambda x: x.split(' ')  
    word_to_id = build_vocab(write_data_path[0], tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
    pkl.dump(word_to_id, open(write_data_path[3], 'wb'))

    embeddings = np.random.rand(len(word_to_id), emb_dim)
    #　加载pretrain word vectors
    if os.path.exists(write_data_path[5]):
        with open(write_data_path[5], "r", encoding='UTF-8') as f:
            for i, line in enumerate(f.readlines()):
                lin = line.strip().split(" ")
                if lin[0] in word_to_id:
                    idx = word_to_id[lin[0]]
                    emb = [float(x) for x in lin[1:emb_dim+1]]
                    embeddings[idx] = np.asarray(emb, dtype='float32')
    np.savez_compressed(write_data_path[4], embeddings=embeddings)
