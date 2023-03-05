# coding='utf-8'
from time import time
from turtle import left
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
from collections import Counter


def plot_embedding(data, label, title):
    x_min,x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig= plt.figure(dpi=400,figsize=(8,6))
    print(Counter(label))

    for i in range(data.shape[0]):
        if len(np.unique(label)) > 9 :
            plt.text(data[i, 0], data[i, 1], str('*'),
                    color=plt.cm.Set3((label[i])))
        else:
            plt.text(data[i, 0], data[i, 1], str('*'),
                    color=plt.cm.Set1((label[i])))
        
    # plt.xticks([])
    # plt.yticks([])
    # plt.title(title)
    fig = plt.gcf()
    fig.subplots_adjust(left=0.1, bottom=0.1)

    time_str = str(time())
    
    ax=plt.gca()  #gca:get current axis得到当前轴
    #设置图片的右边框和上边框为不显示
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    plt.axis('off')

    # plt.legend()
    plt.savefig(title+".png", dpi=600)
    print('plot success!!')
    # plt.show()

def plot_embedding_3d(X,y, title=None):
    #坐标缩放到[0,1]区间
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)
    #降维后的坐标为（X[i, 0], X[i, 1],X[i,2]），在该位置画出对应的
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    for i in range(X.shape[0]):
        ax.text(X[i, 0], X[i, 1], X[i,2], str('^'), color=plt.cm.Set1(y[i] / 10.))

    # plt.title(title)
    time_str = str(time())
    plt.savefig(title + ".png")
    # plt.show()


def main(data,label,title):
    print("Computing t-SNE embedding")
    tsne = TSNE(n_components=2, random_state=77)
    t_start = time()
    result = tsne.fit_transform(data)
    print("takes time %.2fs" % (time()-t_start))
    plot_embedding(result, label, title)



if __name__ == '__main__':

    pass
    # from datasets import load_data, load_mat
    # x, y, x1, y1 = load_data('Aminer')  # xy为作者标签 #x1y1为摘要维度

    # # # x为数据集的feature，y为label.

    # idx = np.random.choice(np.arange(len(x)),700,replace=False)
    # print(type(x))
    # x_sample = x[idx] + x1[idx]
    # y_sample = y[idx]
    # main(x_sample, y_sample, 'original')




