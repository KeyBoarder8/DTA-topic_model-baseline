import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from configuration import model_name,label_mapping_lv1 ,label_mapping_lv3,learning_rate
from torch.autograd import Variable
from utils import predictions_analysis,maybe_cuda
from tqdm import tqdm
import random

def import_model(model_name,label_mapping):
    module = __import__('' + model_name, fromlist=[''])
    return module.create(label_mapping)

def train(model, epoch, dataset, optimizer):
    model.train()
    total_loss = float(0)
    for i, (data, target) in tqdm(enumerate(dataset)):
        model.zero_grad()
        target_var = Variable(torch.cat(target, 0), requires_grad=False)
        loss = model.neg_log_likelihood(data, target_var)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()


    total_loss = total_loss / len(dataset)
    print('Training Epoch: {}, Loss: {:.4}.'.format(epoch + 1, total_loss))




def validate(model, epoch, dataset):
    model.eval()
    with tqdm(desc='Validatinging', total=len(dataset)) as pbar:
        preds_stats = predictions_analysis()
        for i, (data, target) in enumerate(dataset):
            pbar.update()
            output = model(data)
            targets_var = Variable(torch.cat(target, 0), requires_grad=False)
            target_seg = targets_var.data.cpu().numpy()
            preds_stats.add(output, target_seg)

    epoch_f1 = preds_stats.get_micro_f1()
    return epoch_f1


def epoch_train(epoch_num,model, optimizer,train_dl,val_dl,best_model_out_path):
    best_res = 0.0 
    for j in range(epoch_num):
        train(model, j, train_dl, optimizer)
        val_f1 = validate(model, j, val_dl)
        if val_f1 > best_res:
            print('model saving in: ' + best_model_out_path)
            torch.save(model.state_dict(), best_model_out_path)
            best_res = val_f1
            print('best val_f1:',val_f1)

model_lv1 = maybe_cuda(import_model(model_name,label_mapping_lv1))
model_lv3 = maybe_cuda(import_model(model_name,label_mapping_lv3))

optimizer_lv1 = torch.optim.AdamW(model_lv1.parameters(), lr=learning_rate, weight_decay=5e-4)
optimizer_lv3 = torch.optim.AdamW(model_lv3.parameters(), lr=learning_rate, weight_decay=5e-4)