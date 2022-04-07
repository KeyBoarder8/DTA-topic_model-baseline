from shutil import copy
from sklearn.metrics import f1_score
import torch

def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()


def maybe_cuda(x):
    if torch.cuda.is_available():
        return x.cuda()
    return x


def unsort(sort_order):
    result = [-1] * len(sort_order)

    for i, index in enumerate(sort_order):
        result[index] = i

    return result


class predictions_analysis(object):
    def __init__(self):
        self.new_prd = []
        self.new_target = []
        
    def add(self,predictions, targets):
        for i in range(len(predictions)):
            if predictions[i] % 2 != 0:
                self.new_prd.append(predictions[i]-1)
            else:
                self.new_prd.append(predictions[i])
            if targets[i] % 2 != 0:
                self.new_target.append(targets[i] - 1)
            else:
                self.new_target.append(targets[i])

    def get_micro_f1(self):
        return  f1_score(self.new_target, self.new_prd, average='micro')