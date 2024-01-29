from __future__ import division
from __future__ import print_function
import csv
import time
import argparse
import torch.nn.functional as F
import torch
import torch.optim as optim
import numpy as np
from sklearn.model_selection import StratifiedKFold
from utils.metrics import accuracy
from sklearn.metrics import roc_auc_score,f1_score,recall_score,precision_score,matthews_corrcoef
from model import ResDeepGCN
import torch.nn as nn


parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,help='Validate during training pass.')
parser.add_argument('--bias', action='store_true', default=True,help='bias.')
parser.add_argument('--seed', type=int, default=28, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,help='Number of epochs to train.')
parser.add_argument('--LEARNING_RATE', type=float, default=0.01,help='LEARNING_RATE.')
parser.add_argument('--WEIGHT_DACAY', type=float, default=5e-4,help='WEIGHT_DACAY.')
parser.add_argument('--dropout', type=float, default=0.6, help='dropout.')
act=F.tanh
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


features = np.loadtxt("../../Outputs/X_temp.csv",delimiter=",")
labels = np.loadtxt("../../Outputs/Y_temp.csv",delimiter=",")
adj = np.loadtxt("../../Outputs/fused_network.csv",delimiter=",")

# num_nodes, input_dim=features.shape


j=1;val_acc_mean=0.0;val_auc_mean = 0.0;val_f1_mean = 0.0;val_Pre_mean = 0.0;val_Recall_mean = 0.0;Matthews_val_mean=0.0
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=222)
for idx_train, idx_val,in kfold.split(features, labels):
    print("*****",j, "Fold***************************************************")

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    adj = torch.FloatTensor(adj)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)


    model = ResDeepGCN(in_channels=features.shape[1],channels1=64,channels2=64,n_blocks=3,
                       n_classes=labels.max().item() + 1,act='relu',dropout=args.dropout,norm=None,bias=True,
                       conv='gcn',heads=8)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.LEARNING_RATE, weight_decay=args.WEIGHT_DACAY)


    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = torch.FloatTensor(adj).cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()


    t_total = time.time()
    for epoch in range(args.epochs):
            t = time.time()
            model.train()
            optimizer.zero_grad()
            output,feature_fusion_train = model(features,adj)
            loss_train = criterion(output[idx_train], labels[idx_train])
            acc_train = accuracy(output[idx_train], labels[idx_train])
            if len(np.unique(output[idx_train].max(1)[1].cpu()))!= 2:
                roc_auc_score_train=torch.tensor(0.5)
            else:
                roc_auc_score_train = roc_auc_score(labels[idx_train].cpu().detach().numpy(),output[idx_train].max(1)[1].cpu())
            f1_train = f1_score(labels[idx_train].cpu().detach().numpy(),output[idx_train].max(1)[1].cpu(),zero_division=0)
            precision_train= precision_score(labels[idx_train].cpu().detach().numpy(),output[idx_train].max(1)[1].cpu())
            recall_train = recall_score(labels[idx_train].cpu().detach().numpy(),output[idx_train].max(1)[1].cpu(),zero_division=0)
            matthews_train = matthews_corrcoef(labels[idx_train].cpu().detach().numpy(),output[idx_train].max(1)[1].cpu())

            loss_train.backward()
            optimizer.step()
            if args.fastmode==False:
                model.eval()
                output,feature_fusion_val = model(features,adj)

            loss_val = criterion(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])
            if len(np.unique(output[idx_val].max(1)[1].cpu()))!= 2:
                roc_auc_score_val=torch.tensor(0.5)
            else:
                roc_auc_score_val = roc_auc_score(labels[idx_val].cpu().detach().numpy(),output[idx_val].max(1)[1].cpu())
            f1_val = f1_score(labels[idx_val].cpu().detach().numpy(),output[idx_val].max(1)[1].cpu())
            precision_val = precision_score(labels[idx_val].cpu().detach().numpy(),output[idx_val].max(1)[1].cpu())
            recall_val  = recall_score(labels[idx_val].cpu().detach().numpy(),output[idx_val].max(1)[1].cpu(),zero_division=0)
            matthews_val  = matthews_corrcoef(labels[idx_val].cpu().detach().numpy(),output[idx_val].max(1)[1].cpu())


            print('Epoch: {:04d}'.format(epoch+1),
                    'loss_train: {:.4f}'.format(loss_train.item()),
                    'acc_train: {:.4f}'.format(acc_train.item()),
                    'auc_train: {:.4f}'.format(roc_auc_score_train.item()),
                    'f1_train: {:.4f}'.format(f1_train.item()),
                    'Pre_train: {:.4f}'.format(precision_train.item()),
                    'Recall_train: {:.4f}'.format(recall_train.item()),
                    'Matthews_train: {:.4f}'.format(matthews_train),
                    'loss_val: {:.4f}'.format(loss_val.item()),
                    'acc_val: {:.4f}'.format(acc_val.item()),
                    'auc_val: {:.4f}'.format(roc_auc_score_val.item()),
                    'f1_val: {:.4f}'.format(f1_val.item()),
                    'Pre_val: {:.4f}'.format(precision_val.item()),
                    'Recall_val: {:.4f}'.format(recall_val.item()),
                    'Matthews_val: {:.4f}'.format(matthews_val),
                    'time: {:.4f}s'.format(time.time() - t))

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    j+=1
    features = features.to(device='cpu')
    labels = labels.to(device='cpu')
    adj = adj.to(device='cpu')
    idx_train = idx_train.to(device='cpu')
    idx_val = idx_val.to(device='cpu')




model.eval()
features = torch.FloatTensor(features).to('cuda')
adj = torch.FloatTensor(adj).to('cuda')
output,feature_fusion = model(features,adj)
print(feature_fusion.shape)
np.savetxt('../../Outputs/PSN.csv', feature_fusion.cpu().detach().numpy(), delimiter=',')


print('acc_val_mean:{:.4f}'.format(val_acc_mean/10.0),
      'auc_val_mean:{:.4f}'.format(val_auc_mean/10.0),
      'f1_val_mean:{:.4f}'.format(val_f1_mean/10.0),
      'Pre_val_mean:{:.4f}'.format(val_Pre_mean/10.0),
      'Recall_val_mean:{:.4f}'.format(val_Recall_mean/10.0),
      'Matthews_val_mean:{:.4f}'.format(Matthews_val_mean/10.0))


