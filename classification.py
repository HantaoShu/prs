import pickle as pkl

from torch.utils.data.dataloader import default_collate

from model.model_relu import GCN
from model.model_tanh import GCN as GCN_tanh
from model.model_leaky_relu import GCN as GCN_leaky_relu
from model.model_leaky_relu_2 import GCN as GCN_leaky_relu_2
from model.model_leaky_relu_3 import GCN as GCN_leaky_relu_3
from model.model_leaky_relu_4 import GCN as GCN_leaky_relu_4
from model.model_leaky_relu_5 import GCN as GCN_leaky_relu_5
from model.model_leaky_relu_1_4 import GCN as GCN_leaky_relu_1_4
from model.model_prelu import GCN as GCN_prelu
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import argparse
import warnings
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import random

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()

parser.add_argument("--select", type=str)
parser.add_argument("--k", type=str)
parser.add_argument("--lgraph", type=float,default=0)
parser.add_argument("--feature", type=int,default=0)
parser.add_argument("--dir", type=str,default='heart/HCM')
parser.add_argument("--epoch", type=int,default=350)
parser.add_argument("--type", type=str,default='relu')
args = parser.parse_args()

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

data = pkl.load(open(f'./{args.dir}/{args.select}_{args.k}_data.pkl', 'rb'))
# X,y,edge,edge_weight,_
# data2= pkl.load(open(f'./{args.dir}/ATAC_10330_24feature_edge_{args.k}_data.pkl', 'rb'))
X = data[0]
y = data[1]
edge= data[2]
X[np.isnan(X)]=0
print('before',np.isnan(X).sum())
means = X.mean(0, keepdims=True)
stds = X.std(0, keepdims=True)
stds[np.isnan(stds)]=1
X = (X - means) / (stds + 1e-7)
print(np.isnan(X).sum())
print(X.shape)
X_old = X.copy()
L = torch.FloatTensor(edge)
graph = nx.Graph()
for i in range(X.shape[1]):
    graph.add_edge(i, i)
for i in range(len(edge)):
    graph.add_edge(edge[i,0], edge[i,1])
device = torch.device('cuda')
normalized_laplacian_matrix = torch.FloatTensor(nx.normalized_laplacian_matrix(graph).toarray()).to(device)

y = torch.FloatTensor(y).to(device)
L = torch.FloatTensor(L).long().to(device).T
# edge_weight = torch.FloatTensor(edge_weight).to(device)
edge_weight = torch.ones(L.shape[1]).to(device)
aucs = []
for sample in range(1):
    for layer in tqdm([4,0,1,2,3]):
        for l1 in tqdm([1,10,50,100,250,500,750]):
            for lgraph in tqdm([args.lgraph]):
                for l0 in ([0]):
                    for random_state in tqdm(range(5)):
                        X = torch.FloatTensor(X_old).to(device)
                        # X = X[:, :, args.feature:args.feature+1]
                        print('!!!!',X.shape)
                        kf = KFold(5, shuffle=True, random_state=random_state)
                        for i, (train_ind, test_ind) in (enumerate(kf.split(X, y))):
                            train_ind, val_ind = train_test_split(train_ind, random_state=random_state, test_size=0.25)
                            X_train = (X[train_ind])
                            y_train = (y[train_ind])
                            y_train_np= y_train.detach().cpu().numpy()
                            X_val = (X[val_ind])
                            y_val = (y[val_ind])
                            X_test = (X[test_ind])
                            y_test = (y[test_ind])
                            if args.type =='relu':
                                model = GCN(X.shape[-1], X.shape[1], layer, 1).to(device)
                            elif args.type == 'leaky_relu':
                                model = GCN_leaky_relu(X.shape[-1],X.shape[1],layer,1).to(device)
                            elif args.type == 'leaky_relu_0.2':
                                model = GCN_leaky_relu_2(X.shape[-1],X.shape[1],layer,1).to(device)
                            elif args.type == 'leaky_relu_0.2_V2':
                                model = GCN_leaky_relu_3(X.shape[-1],X.shape[1],layer,1).to(device)
                            elif args.type == 'leaky_relu_0.2_V3':
                                model = GCN_leaky_relu_4(X.shape[-1],X.shape[1],layer,1).to(device)
                            elif args.type == 'leaky_relu_0.1_V3':
                                model = GCN_leaky_relu_1_4(X.shape[-1],X.shape[1],layer,1).to(device)
                            elif args.type == 'leaky_relu_0.2_V4':
                                model = GCN_leaky_relu_5(X.shape[-1],X.shape[1],layer,1).to(device)
                            elif args.type == 'prelu':
                                model = GCN_prelu(X.shape[-1],X.shape[1],layer,1).to(device)
                            elif args.type == 'tanh':
                                model = GCN_tanh(X.shape[-1], X.shape[1], layer, 1).to(device)
                            optim = torch.optim.Adam(model.parameters(), lr=1e-3)
                            n_epoch = args.epoch
                            # dataset = TensorDataset(X_train, y_train)
                            # dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True, collate_fn=
                            #     lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
                            for epoch in tqdm(range(n_epoch)):
                                y_id = pd.DataFrame(y_train_np).reset_index()
                                y_id_pos = y_id[y_id[0] == 1].sample(n=len(y_id[0]) - int(sum(y_id[0])), replace=True,random_state=epoch)
                                select_idn = np.array(list(y_id_pos['index']) + list(y_id[y_id[0] == 0]['index']))
                                dataset = TensorDataset(X_train[select_idn], y_train[select_idn])
                                # print('aaa',len(select_idn),len(test_ind),len(val_ind),len(train_ind),'bbb')
                                dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True,)
                                # collate_fn = lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
                                for xx, yy in dataloader:
                                    optim.zero_grad()
                                    out = model(xx, L, edge_weight)[:, 0]
                                    loss = F.binary_cross_entropy_with_logits(out, yy)
                                    loss2 = abs(model.pred.weight).mean() * l0 + (model.pred.weight ** 2).mean() * l1  #
                                    loss3 = model.pred.weight @ normalized_laplacian_matrix @ model.pred.weight.T * lgraph \
                                            / len(model.pred.weight)
                                    loss = loss + loss2 + loss3
                                    # print(loss)
                                    loss.backward()
                                    optim.step()
                                if epoch % 1 == 0:
                                    with torch.no_grad():
                                        model.eval()
                                        out = model(X_test, L, edge_weight)[:, 0]
                                        aucs.append([args.feature, sample, l0, l1, lgraph, layer, i, epoch,
                                                     roc_auc_score(y_test.cpu().detach().numpy(), out.cpu().detach().numpy())])
                                        out = model(X_val, L, edge_weight)[:, 0]
                                        aucs[-1].append(roc_auc_score(y_val.cpu().detach().numpy(), out.cpu().detach().numpy()))

                                        out = model(X_train, L, edge_weight)[:, 0]
                                        aucs[-1].append(roc_auc_score(y_train.cpu().detach().numpy(), out.cpu().detach().numpy()))
                                        aucs[-1].append(random_state)
                                        # print(aucs[-1])
                    type_name = args.dir.replace('/','')
                    pd.DataFrame(aucs).to_csv(f'performance/fauc_abs{args.select}_{args.k}_{args.feature}_{type_name}_'+
                                              f'edge1_V3_nograph_lgarph_{args.lgraph}_V4_{args.type}_tmp1108_relu.csv')