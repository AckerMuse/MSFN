import numpy as np
import pandas as pd
import snf
import random
random.seed(2)

clin=pd.read_csv("../../Data/Clin_33.csv", index_col=False, header=None).astype(float)
cnv=pd.read_csv("../../Data/CNV_200.csv", index_col=False, header=None).astype(float)
mrna=pd.read_csv("../../Data/mRNA_400.csv", index_col=False, header=None).astype(float)
lables=pd.read_csv("../../Data/LABLES.csv", index_col=0, header=0).astype(float)
X_temp= np.concatenate((clin,cnv,mrna),axis=1)

np.savetxt("../../Outputs/X_temp.csv",X_temp,delimiter=",")
np.savetxt("../../Outputs/Y_temp.csv",lables,delimiter=",")
temp = [clin,cnv,mrna]

fused_network = snf.make_affinity(temp, metric='sqeuclidean', K=20, mu=0.5)
fused_network = snf.snf(fused_network, K=20)
np.savetxt("../../Outputs/fused_network.csv",fused_network,delimiter=",")