import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import time
import pandas  as pd

#Calculate the distance function
def distances(x,y):
    dis = (x-y)**2
    distance = dis.sum(axis=0)
    return distance**0.5

#Calculate the distance matrix
def get_dist(data):
    dist = euclidean_distances(data, data)
    indexDistanceAsc = np.argsort(dist)
    return dist,indexDistanceAsc
  
#Calculate the average k-nearest neighbor value for all samples
def k_meandist(data,dist,indenDistanceAsc,k):
    n = len(data)
    k_meandist = np.zeros(n)
    for i in range(n):
        sum = 0
        for j in indenDistanceAsc[i, :k]:
            sum += (dist[i,j]/k)
        k_meandist[i] = sum
    return k_meandist
  
#Calculate the mutual K-nearest neighbor value for all samples and Divide all samples into core and non-core points
def get_reveseknn(indenDistanceAsc,k):
    n=len(dist)
    reverse=np.zeros(n)
    for i in range(n):
        index=0
        for j in indenDistanceAsc[i,:k]:
                if i in indenDistanceAsc[j,:k]:index+=1
        reverse[i]=index
        
    batch1 = []   #Representing core points
    batch2 = []   #Representing non-core points

    mean= int(np.mean(reverse))
    for i in range(len(reverse)):
        if reverse[i] >= mean: batch1.append(i)
        else: batch2.append(i)

    return reverse,batch1,batch2,

#clustering the corepoints
def fist_cluster(batch,k_meandist,dist,indexDistanceAsc,data):
    n=len(batch)
    batch_k_meandist=np.zeros(n)
    for i,x in enumerate(batch):
        batch_k_meandist[i]=k_meandist[x]
    sort=np.argsort(batch_k_meandist)
    batch=np.array(batch)
    new_batch=batch[sort]
    new_batch=list(new_batch)
    cluster = []
    center = []
    cluster.append([new_batch[0]])
    center.append(new_batch[0])
    i = 0
    while len(new_batch)!=0:
        flag = 0
        # print(i)
        for j in  indexDistanceAsc[cluster[i][0],:]:
            if j in new_batch:
               if np.min(dist[j][cluster[i]]) < max(np.max(k_meandist[cluster[i]]),k_meandist[j]):
                  cluster[i].append(j)
                  new_batch.remove(j)
                  flag=1
        if flag==0:
            if len(new_batch)!=0:
                center.append(new_batch[0])
                cluster.append([new_batch[0]])
                new_batch.remove(new_batch[0])
                i+=1

    return cluster, center
  
#Indexed sorting based on density for all non-core points

def sortbatch2(batch,k_meandist):
    batch = np.array(batch)
    batchdensity=np.zeros(len(batch))
    for i in range(len(batch)):
        batchdensity[i]=k_meandist[batch[i]]
    sort=np.argsort(batchdensity)
    newbatch=batch[sort]
    return newbatch

#Allocation of remaining non-core points.
def secondstep(data,batch,cluster,indexDistanceAsc,k_meandist):

    index = np.full(len(data), -1)
    for i in range(len(cluster)):
        for j in cluster[i]:
            index[j] = i
    for x in batch:
        for j in indexDistanceAsc[int(x), :]:
               if index[int(j)] != -1:
                   index[int(x)]=index[int(j)]
                   break

    return index
  
  if __name__ == '__main__':

    import perpy as py
    x,r= py.load(path=r'   ',col_labels=2)
    #x, r = py.load(path=r'   ', col_labels=0)
    #x= py.load(path=r'   ', col_labels=None)
    # r = py.load(path=r'   ', col_labels=None,scaling=False)
    print(x)
    txt=['aggregation.txt','jain.txt','spiral.txt','flame.txt','Twomoons.txt','cth3.txt','d6.txt','Zigzag.txt','threecircles.txt','pasepathed.txt','t4.txt']
    k1=[11,17,10,11,18,15,55,35,11,17,20,40]
    k2=[11,17,10,11,18,15,35,35,11,12,70,70]

    dist, indexDistanceAsc = get_dist1(x)  #n2
    k_meandist = k_meandist(x, dist, indexDistanceAsc, k1[10])   #kn
    reverse, batch1, batch2 = get_reveseknn(indexDistanceAsc, k2[10])#kn
    
    #Visualisation of core and non-core points
    for i in batch1:
        plt.plot(x[i, 0], x[i, 1], marker='o',color='salmon', markersize=4)
    for i in batch2:
        plt.plot(x[i, 0], x[i, 1], marker='3',color='k', markersize=10)
    plt.show()

    batch2=sortbatch2(batch2,k_meandist)  #nlogn
    cluter, center=fist_cluster(batch1,k_meandist,dist,indexDistanceAsc,x)
    index=secondstep(x,batch2,cluter,indexDistanceAsc,k_meandist)
    py.plt_scatter(x,labels=index,center=center)
