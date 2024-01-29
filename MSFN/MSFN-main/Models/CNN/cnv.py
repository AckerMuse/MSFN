from __future__ import division
from __future__ import print_function
import math
import pandas as pd
import argparse
import torch
import torch.optim as optim
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score,f1_score,recall_score,precision_score,matthews_corrcoef,accuracy_score
from model import CNN_CNV
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

CUDA_LAUNCH_BLOCKING=1
comment="CNV"
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,help='Validate during training pass.')
parser.add_argument('--bias', action='store_true', default=True,help='bias.')
parser.add_argument('--seed', type=int, default=28, help='Random seed.')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size.')
parser.add_argument('--epochs', type=int, default=20,help='Number of epochs to train.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]




clin = pd.read_csv("../../Data/CNV_200.csv", index_col=False, header=None).astype('float').values
labels= pd.read_csv("../../Data/LABLES.csv", index_col=0, header=0).values


#10折交叉验证
j=1;val_acc_mean=0.0;val_auc_mean = 0.0;val_f1_mean = 0.0;val_Pre_mean = 0.0;val_Recall_mean = 0.0;Matthews_val_mean=0.0
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=222)
for idx_train, idx_val,in kfold.split(clin, labels):
    print("*****",j, "Fold***************************************************")
    j += 1
    clin = torch.FloatTensor(clin)
    labels = torch.FloatTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)

    x_train_clinical, y_train_clinical = clin[idx_train], labels[idx_train]
    x_test_clinical, y_test_clinical = clin[idx_val], labels[idx_val]
    x_train_clinical = np.expand_dims(x_train_clinical, axis=1)
    x_test_clinical = np.expand_dims(x_test_clinical, axis=1)
    x_train_clinical = torch.FloatTensor(x_train_clinical)
    x_test_clinical = torch.FloatTensor(x_test_clinical)


    #加载模型
    model = CNN_CNV()
    model.init_weights()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, amsgrad=False,weight_decay=1e-2)

    #调用gpu
    if args.cuda:
        model.cuda()
        clin = clin.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        x_train_clinical=x_train_clinical.cuda()
        x_test_clinical =x_test_clinical.cuda()
        y_train_clinical=y_train_clinical.cuda()
        y_test_clinical =y_test_clinical.cuda()

    #训练
    for epoch in range(args.epochs):

        dataset = MyDataset(x_train_clinical, y_train_clinical)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        loss_total=0.0;acc_total=0.0;auc_total=0.0;f1_total=0.0;pre_total=0.0;recall_total=0.0;matthews_total=0.0
        for batch_data, batch_labels in dataloader:

            model.train()
            optimizer.zero_grad()
            output,_ = model(batch_data)

            loss_train = criterion(output, batch_labels.reshape(-1,1))
            acc_train = accuracy_score(batch_labels.cpu().detach().numpy(),torch.round(output).cpu().detach().numpy())
            if len(np.unique(batch_labels.cpu().detach().numpy())) != 2:
                roc_auc_score_train = torch.tensor(0.5)
            else:
                roc_auc_score_train = roc_auc_score(batch_labels.cpu().detach().numpy(),torch.round(output).cpu().detach().numpy())
            f1_train = f1_score(batch_labels.cpu().detach().numpy(),torch.round(output).cpu().detach().numpy(),zero_division=0)
            precision_train= precision_score(batch_labels.cpu().detach().numpy(),torch.round(output).cpu().detach().numpy(),zero_division=0)
            recall_train = recall_score(batch_labels.cpu().detach().numpy(),torch.round(output).cpu().detach().numpy(),zero_division=0)
            matthews_train = matthews_corrcoef(batch_labels.cpu().detach().numpy(),torch.round(output).cpu().detach().numpy())

            loss_train.backward()
            optimizer.step()

            loss_total   =loss_total+loss_train
            acc_total    =acc_total+acc_train
            auc_total    = auc_total + roc_auc_score_train
            f1_total     = f1_total + f1_train
            pre_total    = pre_total + precision_train
            recall_total = recall_total + recall_train
            matthews_total = matthews_total + matthews_train





        loss_train=loss_total/math.ceil(float(x_train_clinical.shape[0])/float(args.batch_size))
        acc_train = acc_total/math.ceil(float(x_train_clinical.shape[0])/float(args.batch_size))
        roc_auc_score_train = auc_total / math.ceil(float(x_train_clinical.shape[0]) / float(args.batch_size))
        f1_train = f1_total / math.ceil(float(x_train_clinical.shape[0]) / float(args.batch_size))
        pre_total= precision_train / math.ceil(float(x_train_clinical.shape[0]) / float(args.batch_size))
        recall_train = recall_total / math.ceil(float(x_train_clinical.shape[0]) / float(args.batch_size))
        matthews_train = matthews_total / math.ceil(float(x_train_clinical.shape[0]) / float(args.batch_size))



        #评估模型
        model.eval()
        output,_  = model(x_test_clinical)
        loss_val = criterion(output, y_test_clinical.reshape(-1,1))
        acc_val = accuracy_score(y_test_clinical.cpu().detach().numpy(),torch.round(output).cpu().detach().numpy())
        if len(np.unique(batch_labels.cpu().detach().numpy())) != 2:
            roc_auc_score_val = torch.tensor(0.5)
        else:
            roc_auc_score_val = roc_auc_score(y_test_clinical.cpu().detach().numpy(),torch.round(output).cpu().detach().numpy())
        f1_val = f1_score(y_test_clinical.cpu().detach().numpy(),torch.round(output).cpu().detach().numpy(),zero_division=0)
        precision_val = precision_score(y_test_clinical.cpu().detach().numpy(),torch.round(output).cpu().detach().numpy(),zero_division=0)
        recall_val = recall_score(y_test_clinical.cpu().detach().numpy(),torch.round(output).cpu().detach().numpy(),zero_division=0)
        matthews_val = matthews_corrcoef(y_test_clinical.cpu().detach().numpy(),torch.round(output).cpu().detach().numpy())
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train),
              'acc_train: {:.4f}'.format(acc_train),
              'auc_train: {:.4f}'.format(roc_auc_score_train),
              'f1_train: {:.4f}'.format(f1_train),
              'Pre_train: {:.4f}'.format(precision_train),
              'Recall_train: {:.4f}'.format(recall_train),
              'Matthews_train: {:.4f}'.format(matthews_train),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'auc_val: {:.4f}'.format(roc_auc_score_val.item()),
              'f1_val: {:.4f}'.format(f1_val.item()),
              'Pre_val: {:.4f}'.format(precision_val.item()),
              'Recall_val: {:.4f}'.format(recall_val.item()),
              'Matthews_val: {:.4f}'.format(matthews_val))
        if epoch == args.epochs-1:
            val_acc_mean = float(val_acc_mean + acc_val)
            val_auc_mean = float(val_auc_mean + roc_auc_score_val)
            val_f1_mean = float(val_f1_mean + f1_val)
            val_Pre_mean = float(val_Pre_mean + precision_val)
            val_Recall_mean = float(val_Recall_mean + recall_val)
            Matthews_val_mean = float(Matthews_val_mean + matthews_val)
    clin = clin.to(device='cpu')
    labels = labels.to(device='cpu')
    idx_train = idx_train.to(device='cpu')
    idx_val = idx_val.to(device='cpu')


X_clinical = np.expand_dims(clin, axis=1)
X_clinical = torch.FloatTensor(X_clinical).to('cuda')
_,stacked_feature = model(X_clinical)
print(stacked_feature.shape)
stacked_feature=stacked_feature.cpu().detach().numpy()
np.savetxt("../../Outputs/CNV_feature.csv",stacked_feature,delimiter=',')

print('acc_val_mean:{:.4f}'.format(val_acc_mean/10.0),
      'auc_val_mean:{:.4f}'.format(val_auc_mean/10.0),
      'f1_val_mean:{:.4f}'.format(val_f1_mean/10.0),
      'Pre_val_mean:{:.4f}'.format(val_Pre_mean/10.0),
      'Recall_val_mean:{:.4f}'.format(val_Recall_mean/10.0),
      'Matthews_val_mean:{:.4f}'.format(Matthews_val_mean/10.0))
