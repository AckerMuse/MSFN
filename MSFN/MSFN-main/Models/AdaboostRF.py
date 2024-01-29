from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import matthews_corrcoef,make_scorer
import random
random.seed(1)

def mcc_scorer(y_true, y_pred):
    return matthews_corrcoef(y_true, y_pred)



data1=pd.read_csv("../../Outputs/CLin_feature.csv",header=None,index_col=False)
data2=pd.read_csv("../../Outputs/CNV_feature.csv",header=None,index_col=False)
data3=pd.read_csv("../../Outputs/mRNA_feature.csv",header=None,index_col=False)
data4=pd.read_csv("../../Outputs/PSN_feature.csv",header=None,index_col=False)
Y1=pd.read_csv("../../Data/LABLES.csv", index_col=0, header=0).astype(float)
Y1 = np.ravel(Y1)
X1 =pd.concat([data1,data2,data3,data4], axis=1)


rf= RandomForestClassifier(n_estimators=200, max_depth=None, random_state=0,class_weight='balanced')
rf = AdaBoostClassifier(base_estimator=rf,n_estimators=200)

scores1 = cross_val_score(rf, X1, Y1, cv=10,verbose=0)
scores_precision = cross_val_score(rf, X1, Y1, cv=10, scoring='precision', verbose=0)
scores_recall = cross_val_score(rf, X1, Y1, cv=10, scoring='recall', verbose=0)
scores_f1 = cross_val_score(rf, X1, Y1, cv=10, scoring='f1', verbose=0)
scores_auc = cross_val_score(rf, X1, Y1, cv=10, scoring='roc_auc', verbose=0)
mcc_scorer = make_scorer(mcc_scorer)
scores_mcc = cross_val_score(rf, X1, Y1, cv=10, scoring=mcc_scorer, verbose=0)
print ("Cross-validated scores:", scores1)
print("Accuracy = %.3f%% (+/- %.3f%%)" % (np.mean(scores1), np.std(scores1)))
print("AUC = %.3f%% (+/- %.3f%%)" % (np.mean(scores_auc), np.std(scores_auc)))
print("Precision = %.3f%% (+/- %.3f%%)" % (np.mean(scores_precision), np.std(scores_precision)))
print("Recall = %.3f%% (+/- %.3f%%)" % (np.mean(scores_recall), np.std(scores_recall)))
print("F1 Score = %.3f%% (+/- %.3f%%)" % (np.mean(scores_f1), np.std(scores_f1)))
print("MCC = %.3f (+/- %.3f)" % (np.mean(scores_mcc), np.std(scores_mcc)))

