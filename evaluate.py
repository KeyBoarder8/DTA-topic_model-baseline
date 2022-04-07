import torch
from configuration import *
from utils import maybe_cuda
from train import *
import numpy as np
from tqdm import tqdm
from dataloader import *
import jieba_fast as jieba
from itertools import groupby

def import_model(model_name, label_mapping):
    module = __import__('' + model_name, fromlist=[''])
    return module.create(label_mapping)


def eval_data_read(dialog_all):
    dialogs_info = dialog_all['dialogues']
    text_info = [i['text'] for i in dialog_all['dialogues']]
    return text_info, dialogs_info

def json2list(data_path):
    rid_dialogs, rid_text = [],[]
    with open(data_path,'r') as f:
        for line in f:
            dialog_all = eval(line)
            text_info, dialogs_info = eval_data_read(dialog_all)
            rid_dialogs.append(dialogs_info)
            rid_text.append(text_info)
    return rid_dialogs, rid_text


def text_transform(dialog, vocab, w2v):
    sen_vec = []
    for sen in dialog:
        token = list(jieba.cut(sen))
        words_line = []
        for word in token:
            words_line.append(vocab.get(word, vocab.get('<UNK>')))
        if len(words_line) == 0:
            words_line.append(vocab.get('<UNK>'))
        for tok in range(len(words_line)):
            words_line[tok] = w2v[words_line[tok]]
        sen_vec.append(torch.FloatTensor(np.array(words_line)))
    return sen_vec

def topic_gen(rid_text_list, label_mapping, vocab_path, word_vec_path, best_model_out_path):
    vocab = pkl.load(open(vocab_path, 'rb'))
    w2v = np.load(word_vec_path)['embeddings']
    doc_vec_all = []
    for dialog in rid_text_list:
        sen_vec = text_transform(dialog, vocab, w2v)
        doc_vec_all.append(sen_vec)

    model = maybe_cuda(import_model(model_name,label_mapping))
    model.load_state_dict(torch.load(best_model_out_path))
    model.eval()
    out = []
    for doc in tqdm(doc_vec_all):
        out.append(model([doc]))
    one_theme = []
    all_theme = []
    dict_new2 = dict(zip(label_mapping.values(), label_mapping.keys()))
    for doc_out in out:
        for predic in doc_out:
            if predic % 2 == 0:
                one_theme.append(dict_new2[predic + 1].split('_')[1])
            else:
                one_theme.append(dict_new2[predic].split('_')[1])
        all_theme.append(one_theme)
        one_theme = []
    return all_theme

def num2label(mapping):
    mapping_new = {}
    for k,v in mapping.items():
        mapping_new[v] = k
    return mapping_new

def topiclist_gen(theme_list, flag):
    topics_result = []
    cnt = 0
    for k,v in groupby(theme_list):
        num = len(list(v))
        topic_item = {}
        if flag == 1:
            topic_item["topic_level"] = 1
        else:
            topic_item["topic_level"] = 2
        topic_item["start_idx"] = cnt
        topic_item["end_idx"] = cnt+num-1
        topic_item["topic_role_type"] = k
        cnt += num
        topics_result.append(topic_item)
    
    return topics_result

def submit_result(all_theme_lv1, all_theme_lv3, rid_dialogs_all,submit_path): 
    write_lines = []
    for i in range(len(rid_dialogs_all)):
        rid_theme = {}
        rid_theme["dialogues"] = rid_dialogs_all[i]
        rid_theme["summary"] = ""
        rid_theme["topics"] = topiclist_gen(all_theme_lv1[i], 1)
        rid_theme["topics"] += topiclist_gen(all_theme_lv3[i], 3)
        write_lines.append(str(rid_theme)+ '\n')
    with open(submit_path, 'w+') as f:    
        f.writelines(write_lines)

    
if __name__ == '__main__':
    rid_dialogs_all,rid_text_all = json2list(origin_test_path)
    all_theme_lv1 = topic_gen(rid_text_all, label_mapping_lv1, write_data_path[3], write_data_path[4], best_model_out_path_lv1)
    all_theme_lv3 = topic_gen(rid_text_all, label_mapping_lv3, write_data_path[3], write_data_path[4], best_model_out_path_lv3)
    submit_result(all_theme_lv1,all_theme_lv3, rid_dialogs_all,submit_path)