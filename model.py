from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import predictions_analysis,maybe_cuda,unsort
from configuration import *
from dataloader import *

preds_stats = predictions_analysis()

class SentenceEncodingRNN(nn.Module):
    def __init__(self, input_size, hidden, num_layers):
        super(SentenceEncodingRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden = hidden
        self.input_size = input_size

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden,
                            num_layers=self.num_layers,
                            dropout=0,
                            bidirectional=True)

    def forward(self, x):
        batch_size = x.batch_sizes[0]
        _, (hidden, _) = self.lstm(x)
        transposed = hidden.transpose(0, 1)  # (batch_size, 4, 128)
        reshaped = transposed.contiguous().view(batch_size, -1)
        return reshaped

class Model(nn.Module):
    def __init__(self, tag_to_ix, sentence_encoder, hidden, num_layers):
        super(Model, self).__init__()
        self.sentence_encoder = sentence_encoder
        self.sentence_lstm = nn.LSTM(input_size=sentence_encoder.hidden * sentence_encoder.num_layers * 2,
                                     hidden_size=hidden,
                                     num_layers=num_layers,
                                     batch_first=True,
                                     dropout=0,
                                     bidirectional=True)

        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix) + 2
        self.START_TAG = self.tagset_size - 2
        self.STOP_TAG = self.tagset_size - 1
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size) 
        )
        self.h2s = nn.Linear(hidden * 2, self.tagset_size)
        self.num_layers = num_layers
        self.hidden = hidden
        self.criterion = nn.CrossEntropyLoss()
        self.transitions.data[self.START_TAG, :] = -10000
        self.transitions.data[:, self.STOP_TAG] = -10000        


    def pad(self, s, max_length):
        s_length = s.size()[0]
        v = s.unsqueeze(0).unsqueeze(0)
        padded = F.pad(v, (0, 0, 0, max_length - s_length))  # (1, 1, max_length, 300)
        shape = padded.size()
        return padded.view(shape[2], 1, shape[3])  # (max_length, 1, 300)

    def pad_document(self, d, max_document_length):
        d_length = d.size()[0]
        v = d.unsqueeze(0).unsqueeze(0)
        padded = F.pad(v, (0, 0, 0, max_document_length - d_length))  # (1, 1, max_length, 300)
        shape = padded.size()
        return padded.view(shape[2], 1, shape[3])  # (max_length, 1, 300)

    def _forward_alg(self, feats):
        init_alphas = maybe_cuda(torch.full([self.tagset_size], -10000.))
        init_alphas[self.START_TAG] = 0.
        forward_var_list = []
        forward_var_list.append(init_alphas)
        for feat_index in range(feats.shape[0]):
            gamar_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[1])
            t_r1_k = torch.unsqueeze(feats[feat_index], 0).transpose(0, 1)
            aa = gamar_r_l + t_r1_k + self.transitions 
            forward_var_list.append(torch.logsumexp(aa, dim=1))
        terminal_var = forward_var_list[-1] + self.transitions[self.STOP_TAG] 
        terminal_var = torch.unsqueeze(terminal_var, 0)
        alpha = torch.logsumexp(terminal_var, dim=1)[0]
        return alpha

    def _get_lstm_features(self, batch):
        sentences_per_doc = []
        all_batch_sentences = []
        for document in batch:
            all_batch_sentences.extend(document)
            sentences_per_doc.append(len(document))
        lengths = [s.size()[0] for s in all_batch_sentences]
        sort_order = np.argsort(lengths)[::-1]
        sorted_sentences = [all_batch_sentences[i] for i in sort_order]
        sorted_lengths = [s.size()[0] for s in sorted_sentences]
        max_length = max(lengths)
        padded_sentences = [self.pad(s, max_length) for s in sorted_sentences]
        big_tensor = torch.cat(padded_sentences, 1)  
        packed_tensor = pack_padded_sequence(big_tensor, sorted_lengths)
        packed_tensor = maybe_cuda(packed_tensor)
        encoded_sentences = self.sentence_encoder(packed_tensor)
        unsort_order = maybe_cuda(torch.LongTensor(unsort(sort_order)))
        unsorted_encodings = encoded_sentences.index_select(0, unsort_order)

        index = 0
        encoded_documents = []
        for sentences_count in sentences_per_doc:
            end_index = index + sentences_count
            encoded_documents.append(unsorted_encodings[index: end_index, :])
            index = end_index

        doc_sizes = [doc.size()[0] for doc in encoded_documents]
        max_doc_size = np.max(doc_sizes)
        ordered_document_idx = np.argsort(doc_sizes)[::-1]
        ordered_doc_sizes = sorted(doc_sizes)[::-1]
        ordered_documents = [encoded_documents[idx] for idx in ordered_document_idx]
        padded_docs = [self.pad_document(d, max_doc_size) for d in ordered_documents]
        docs_tensor = torch.cat(padded_docs, 1)
        packed_docs = pack_padded_sequence(docs_tensor, ordered_doc_sizes)
        sentence_lstm_output, _ = self.sentence_lstm(packed_docs)
        padded_x, _ = pad_packed_sequence(sentence_lstm_output)  
        doc_outputs = []
        for i, doc_len in enumerate(ordered_doc_sizes):
            doc_outputs.append(padded_x[0:doc_len, i, :])  
        unsorted_doc_outputs = [doc_outputs[i] for i in unsort(ordered_document_idx)]
        sentence_outputs = torch.cat(unsorted_doc_outputs, 0)

        x = self.h2s(sentence_outputs)
        return x

    def _score_sentence(self, feats, tags):
        score = maybe_cuda(torch.zeros(1))
        tags = torch.cat([torch.tensor([self.START_TAG], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + feat[tags[i + 1]] + \
                    self.transitions[tags[i + 1], tags[i]] 
                     
        score = score + self.transitions[self.STOP_TAG, tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []
        init_vvars = maybe_cuda(torch.full((1, self.tagset_size), -10000.))
        init_vvars[0][self.START_TAG] = 0

        forward_var_list = []
        forward_var_list.append(init_vvars)

        for feat_index in range(feats.shape[0]):
            gamar_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[1])
            gamar_r_l = torch.squeeze(gamar_r_l)
            next_tag_var = gamar_r_l + self.transitions 
            viterbivars_t, bptrs_t = torch.max(next_tag_var, dim=1)

            t_r1_k = torch.unsqueeze(feats[feat_index], 0)
            forward_var_new = torch.unsqueeze(viterbivars_t, 0) + t_r1_k

            forward_var_list.append(forward_var_new)
            backpointers.append(bptrs_t.tolist())

        terminal_var = forward_var_list[-1] + self.transitions[self.STOP_TAG] 
        best_tag_id = torch.argmax(terminal_var).tolist()
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.START_TAG 
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):
        lstm_feats = self._get_lstm_features(sentence)
        _, tag_seq = self._viterbi_decode(lstm_feats)
        return tag_seq


def create(label_mapping):
    sentence_encoder = SentenceEncodingRNN(input_size=input_size,
                                           hidden=sen_hidden_size,
                                           num_layers=num_layers)
    return Model(label_mapping, sentence_encoder, hidden=hidden_size, num_layers=num_layers)
