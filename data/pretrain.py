import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.nn import Linear
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from evaluation import eva

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

        return x_bar, z


class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001 * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def pretrain_ae(model, dataset, y, path, last_nmi=0):
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=1e-3)
    for epoch in range(30):
        # adjust_learning_rate(optimizer, epoch)
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.cuda()

            x_bar, _ = model(x)
            loss = F.mse_loss(x_bar, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            x = torch.Tensor(dataset.x).cuda().float()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            print('{} loss: {}'.format(epoch, loss))           
            kmeans = KMeans(n_clusters=len(np.unique(y)), n_init=20).fit(z.data.cpu().numpy())
            _, nmi, _, _ = eva(y, kmeans.labels_, epoch)
        
        if last_nmi < nmi:
            last_nmi = nmi
            print('current save epoch is ', epoch)
            torch.save(model.state_dict(), path)
    
    return last_nmi


for name in ['dblp', 'hhar', 'usps']:

    # name = 'dblp'
    train = True
    data_dims = {
        'usps':256,
        'hhar':561,
        'reut':2000,
        'acm':1870,
        'dblp':334,
        'cite':3703,
        'bbc':10000
    }
    print(name)
    input_dim = data_dims[name]

    for embedding_z in [5]:
        dims = [500, 500, 2000, embedding_z] 
        # 10 20 50 100

        # dims = [600, 200, 2000, 10]
        # dims = [500, 500, 2000, 10]
        # embedding_z = dims

        x = np.loadtxt('{}.txt'.format(name), dtype=float)
        y = np.loadtxt('{}_label.txt'.format(name), dtype=int)
        if dims[0] == 500:
            path = './ab_study/embedding_size/{}_{}.pkl'.format(name, str(dims[-1]))
        else:
            path = './ab_study/layers/{}_{}.pkl'.format(name, str(dims[-1]))
        # print(path)
        dataset = LoadDataset(x)

        if not train:
            model = AE(
                n_enc_1=dims[0],
                n_enc_2=dims[1],
                n_enc_3=dims[2],
                n_dec_1=dims[2],
                n_dec_2=dims[1],
                n_dec_3=dims[0],
                n_input=input_dim,
                n_z=dims[-1],).cuda()

            model.load_state_dict(torch.load(path))
            model.eval()
            x = torch.Tensor(dataset.x).cuda().float()
            x_bar, z = model(x)
            kmeans = KMeans(n_clusters=len(np.unique(y)), n_init=20).fit(z.data.cpu().numpy())
            eva(y, kmeans.labels_, 'test of embedding_z={}'.format(embedding_z))
        else:
            last_nmi = 0
            for _ in range(20):
                model = AE(
                    n_enc_1=dims[0],
                    n_enc_2=dims[1],
                    n_enc_3=dims[2],
                    n_dec_1=dims[2],
                    n_dec_2=dims[1],
                    n_dec_3=dims[0],
                    n_input=input_dim,
                    n_z=dims[-1],).cuda()
                nmi = pretrain_ae(model, dataset, y, path, last_nmi)
                last_nmi = nmi
            print('best_nmi={} for embedding_z={}'.format(last_nmi, embedding_z))