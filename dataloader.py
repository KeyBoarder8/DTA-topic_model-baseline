import torch
from torch.utils.data import Dataset
import numpy as np
import pickle as pkl
from tqdm import tqdm
import torch.utils.data as Data
from configuration import *

def collate_fn(batch):
    batched_data = []
    batched_targets = []
    for data, targets in batch:
        tensored_targets = torch.LongTensor(targets)
        batched_targets.append(tensored_targets)
        batched_data.append([torch.FloatTensor(i) for i in data])
    return batched_data, batched_targets

def load_dataset(path,word_vec_path,vocab_path,flag, pad_size=32):
    documents_texts = []
    documents_labels = []
    seq_len_avg = []
    text_line = []
    label_line = []
    vocab = pkl.load(open(vocab_path, 'rb'))
    tokenizer = lambda x: x.split(' ')
    r = np.load(word_vec_path)
    word_vec = r['embeddings']
    with open(path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                documents_texts.append(text_line)
                documents_labels.append(label_line)
                text_line = []
                label_line = []
            else:
                if flag == 1:
                    content, label, _ = lin.split('\t')
                else:
                    content, _, label = lin.split('\t')
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                seq_len_avg.append(seq_len)
                for word in token:
                     words_line.append(vocab.get(word, vocab.get(UNK)))
                for tok in range(len(words_line)):
                    words_line[tok] = word_vec[words_line[tok]]
                text_line.append(np.array(words_line))
                label_line.append(int(label))

    return documents_texts,documents_labels

def split_data_gen(data_path, wordvec_path, vocab_path ,flag):
    documents_texts_train, documents_labels_train = load_dataset(data_path, wordvec_path, vocab_path ,flag)
    dd_dataset = ChoiDataset(documents_texts_train, documents_labels_train)
    ddl = Data.DataLoader(
        dataset=dd_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
    )
    return ddl

def data_iter(write_data_path, flag):
    train_dl = split_data_gen(write_data_path[1], write_data_path[4], write_data_path[3] ,flag)
    val_dl = split_data_gen(write_data_path[2], write_data_path[4], write_data_path[3] ,flag)
    return train_dl,val_dl


class ChoiDataset(Dataset):
    def __init__(self, documents_texts, documents_label):
        self.documents_texts,self.documents_labels = documents_texts, documents_label

    def __getitem__(self,index):
        return self.documents_texts[index],self.documents_labels[index]

    def __len__(self):
        return len(self.documents_texts)
