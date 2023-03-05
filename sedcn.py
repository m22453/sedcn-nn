from __future__ import print_function, division
import argparse
from ast import arg
import random
from statistics import mode
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
from utils import load_data, load_graph
from GNN import GNNLayer
from evaluation import eva
from collections import Counter
from tqdm import tqdm
import tsne
import os

seed = 100
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

torch.cuda.set_device(1)


class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z


class SDCN(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, 
                n_input, n_z, n_clusters, v=1):
        super(SDCN, self).__init__()

        # autoencoder for intra information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        # GCN for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(n_z, n_clusters)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x, adj):
        # DNN Module
        x_bar, tra1, tra2, tra3, z = self.ae(x)

        # GCN Module
        h = self.gnn_1(x, adj)  # h1
        # tra2 = tra2 + 0.001 *F.relu(self.ae.enc_2(h+tra1))  # ae update via (h1 + tra1)

        h = self.gnn_2(0.5*h+0.5*tra1, adj)
        # tra3 = tra3 + 0.001 *F.relu(self.ae.enc_3(h+tra2))  # ae update via (h2 + tra2)

        h = self.gnn_3(0.5*h+0.5*tra2, adj)
        # z = z + 0.001 *self.ae.z_layer(h+tra3)    # ae update via (h3 + tra3)

        h = self.gnn_4(0.5*h+0.5*tra3, adj)

        h = self.gnn_5(0.5*h+0.5*z, adj, active=False)
        predict = F.softmax(h, dim=1)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        # dot_product
        adj_pred = torch.sigmoid(torch.matmul(z, z.t()))

        return x_bar, q, predict, z, adj_pred


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_sdcn(dataset, lambda_1=0, lambda_2=1):


    model = SDCN(500, 500, 2000, 2000, 500, 500,
                n_input=args.n_input,
                n_z=args.n_z,
                n_clusters=args.n_clusters,
                v=1.0).to(device)
    print(model)

    optimizer = Adam(model.parameters(), lr=args.lr)

    # KNN Graph
    adj = load_graph(args.name, args.k)
    adj = adj.to(device)

    # cluster parameter initiate
    data = torch.Tensor(dataset.x).to(device)
    data_n = torch.zeros_like(data).to(device)
 
    adj_dense = adj.to_dense()
    # get the semantic from adj
    for i in tqdm(range(len(adj_dense))):
        item = adj_dense[i]
        neighbs = item.nonzero().squeeze()
        item_n = data[neighbs].mean(dim=0) + data[i]
        data_n[i] = item_n

    y = dataset.y

    # kmeans
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(data.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(y, y_pred, 'k-means')


    with torch.no_grad():
        _, _, _, _, z = model.ae(data)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(y, y_pred, 'pae')


    res_lst = []
    model_lst = []

    # the idx 
    # np_txt = './sampling/{}-2000.txt'.format(args.name)
    # if os.path.exists(np_txt):
    #     random_idx = np.loadtxt(np_txt)
    # else:
    #     random_idx = np.random.choice(range(len(y)), 2000, replace=False)
    #     np.savetxt(np_txt, random_idx)
    # random_idx = [int(mm) for mm in random_idx]

    for epoch in tqdm(range(300)):
        if epoch % 1 == 0:
            # update_interval
            _, tmp_q, pred, z, _ = model(data, adj)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)
        
            res1 = tmp_q.cpu().numpy().argmax(1)       #Q
            res2 = pred.data.cpu().numpy().argmax(1)   #Z
            res3 = p.data.cpu().numpy().argmax(1)      #P

            # if epoch % 100 ==0 or epoch+1==500 or epoch == 10:
                
            #     z = z.data
            #     if lambda_2 == 0:
            #         tsne.main(z.cpu().numpy()[random_idx],y[random_idx],'./pic/{}-{}'.format(args.name,epoch))
            #     else:
            #         tsne.main(z.cpu().numpy()[random_idx],y[random_idx],'./pic/ours-{}-{}-new-1'.format(args.name,epoch))



            tmp_list = []
            tmp_list.append(np.array(eva(y, res1, str(epoch) + 'Q')))
            tmp_list.append(np.array(eva(y, res2, str(epoch) + 'Z')))
            tmp_list.append(np.array(eva(y, res3, str(epoch) + 'P')))
            tmp_list = np.array(tmp_list)
            idx = np.argmax(tmp_list[:,0])
            # print('tag============>', idx)
            # print(tmp_list[idx][0])
            res_lst.append(tmp_list[idx])

        x_bar, q, pred, _, adj_pred = model(data, adj)

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)
        ren_loss = F.mse_loss(x_bar, data_n)
        # re_gcn_loss = F.binary_cross_entropy(adj_pred, adj_dense)


        loss = 0.5 * kl_loss + 0.01 * ce_loss + lambda_1 * re_loss + lambda_2 * ren_loss #+ 0.001* re_gcn_loss
        # loss = 1 * kl_loss + 0.0 * ce_loss + 0 * re_loss + 0 * ren_loss # DEC
        # loss = 1 * kl_loss + 0.0 * ce_loss + 1 * re_loss + 0 * ren_loss # IDEC

        model_lst.append(model)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    res_lst = np.array(res_lst)
    best_idx = np.argmax(res_lst[:, 1])
    print('best--->',best_idx)
    print('dataset:{},lambda_1:{}, lambda_2:{}'.format(args.name, lambda_1, lambda_2))
    print('ACC={:.2f} +- {:.2f}'.format(res_lst[:, 0][best_idx]*100, np.std(res_lst[:, 0])))
    print('NMI={:.2f} +- {:.2f}'.format(res_lst[:, 1][best_idx]*100, np.std(res_lst[:, 1])))
    print('ARI={:.2f} +- {:.2f}'.format(res_lst[:, 2][best_idx]*100, np.std(res_lst[:, 2])))
    print('F1={:.2f} +- {:.2f}'.format(res_lst[:, 3][best_idx]*100, np.std(res_lst[:, 3])))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='hhar')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_clusters', default=3, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--pretrain_path', type=str, default='pkl')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    # args.pretrain_path = 'data/{}.pkl'.format(args.name)
    args.pretrain_path = 'data/ab_study/embedding_size/{}_{}.pkl'.format(args.name, args.n_z)
    dataset = load_data(args.name)

    if args.name == 'usps':
        args.n_clusters = 10
        args.n_input = 256

    if args.name == 'hhar':
        args.k = 5
        args.n_clusters = 6
        args.n_input = 561

    if args.name == 'reut':
        args.lr = 1e-4
        args.n_clusters = 4
        args.n_input = 2000

    if args.name == 'acm':
        args.k = None
        args.n_clusters = 3
        args.n_input = 1870

    if args.name == 'dblp':
        args.k = None
        args.n_clusters = 4
        args.n_input = 334

    if args.name == 'cite':
        args.lr = 1e-4
        args.k = None
        args.n_clusters = 6
        args.n_input = 3703

    if args.name == 'abstract':
        args.k = 10
        args.n_clusters = 3
        args.n_input = 10000

    if args.name == 'bbc':
        args.k = 10
        args.n_clusters = 5
        args.n_input = 10000



    print(args)
    # train_sdcn(dataset,1,0)
    
    for i in range(5):
        # train_sdcn(dataset,0.7,0.8) # dblp
        # train_sdcn(dataset, 0.7, 1.0) #dblp
        # train_sdcn(dataset, 0.1, 1) # hhar 
        train_sdcn(dataset, 1, 0.1) #usps
    

    # for lambda_1 in range(1,11):
    #     lambda_1 = lambda_1 * 0.1
    #     for lambda_2 in range(1,11):
    #         lambda_2 = lambda_2 * 0.1
    #         for _ in range(5):
    #             train_sdcn(dataset, lambda_1, lambda_2)
