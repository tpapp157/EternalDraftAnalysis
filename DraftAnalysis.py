
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import json
from copy import deepcopy

from umap import UMAP
from hdbscan import HDBSCAN

from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import TfidfTransformer

#%%
head = r'https://eternalwarcry.com/deck-builder?main='
FEpath = 'FE_Sheets'
for _,_,files in os.walk(FEpath):
    break


#%%
raw = []
EWlinks = []
for f in files:
    raw.append(pd.read_excel(os.path.join(FEpath, f), sheet_name='Decklists'))
    EWlinks += list(raw[-1]['Filtered'])


#%%
m = re.compile('\d*-\d*:\d')

cardCode = {}
decks = []
for i in EWlinks:
    if 'eternalwarcry.com/deck-builder' in str(i):
        decks.append([])
        temp = i.split('=')[1].split(';')
        for j in temp:
            if m.match(j)!=None:
                c, n = j.split(':')
                decks[-1].append((c, int(n)))
                if not c in cardCode:
                    cardCode[c] = len(cardCode)


#%%
with open('eternal-cards.json', 'r') as f:
    Cards = json.load(f)

#%%
k = list(cardCode.keys())
F = ['F', 'T', 'J', 'S', 'P']
Influence = {}
for i in Cards:
    if ('SetNumber' in i) and ('EternalID' in i):
        c = str(i['SetNumber'])+'-'+str(i['EternalID'])
        if c in k:
            Influence[c] = np.array([len(i['Influence'].split(f))-1 for f in F])

Colors = np.array([[101, 2, 6],
          [143, 98, 49],
          [58, 103, 39],
          [86, 36, 126],
          [42, 62, 143]])


deck_color = []
deck_inf = []
for i in decks:
    temp = np.zeros(5)
    for c,n in i:
        temp += Influence[c]*n
    deck_inf.append(temp/np.sum(temp))
    deck_color.append(np.matmul(temp/np.sum(temp), Colors))

deck_inf = np.array(deck_inf)
deck_color = np.array(deck_color)


#%%
X0 = np.zeros((len(decks), len(cardCode)))
for j,i in enumerate(decks):
    for c,n in i:
        X0[j, cardCode[c]] = n

X0idf = X0 * np.log(X0.shape[0] / np.sum(X0>0, axis=0))

#0: Defiance
#405 Homecoming
#904: Frontier
#3186: Grodov
#3635: Xulta

X = X0[3635:,:]
sigil = np.argsort(np.sum(X, axis=0))[-5:]
X[:, sigil] = 0
tfidf = TfidfTransformer().fit(X)
Xidf = tfidf.transform(X)
#Xidf = X * np.log(X.shape[0] / np.sum(X>0, axis=0))
#Xidf = np.nan_to_num(Xidf)
temp_color = deck_color[3635:,:]

#%%
N = NMF(n_components=20).fit(Xidf)
c = N.components_
C = N.transform(Xidf)
U = UMAP(n_neighbors=60).fit_transform(C)
plt.scatter(U[:,0],U[:,1],c=temp_color/255*1.5)

factions = np.matmul(C.T, deck_inf[3635:,:])
factions = factions / np.sum(factions, axis=1, keepdims=True)

#%%
H = HDBSCAN(min_cluster_size=3).fit(C)
cent = []
for i in range(H.labels_.max()+1):
    cent.append(np.median(C[H.labels_==i, :], axis=0))
    print(np.sum(H.labels_==i))
cent = np.vstack(cent)


#%%
clinks = []
temp = np.argsort(-c)
k = list(cardCode.keys())
for i in range(temp.shape[0]):
    cards = -np.sort(-c)[i,:]
    cards = temp[i,:np.argmin(np.abs(cards-np.mean(cards)-4*np.std(cards)))]
    print(len(cards))
    cards = ';'.join([k[j]+':1' for j in cards])
    clinks.append(head+cards)
    
#%%
import webbrowser
from time import sleep

for f in clinks:
    webbrowser.open_new_tab(f)
    sleep(1)


#%%
def KLdiv(D, C, T, tfidf, cent, ignore=[]):
    eps = 1e-10
    i = np.sum(D, axis=1, keepdims=True)
    
    c = deepcopy(T.components_)
    c[:, ignore] = 0
    
    
    temp = T.transform(tfidf.transform(D))
    
    clust = np.argmin(np.linalg.norm(np.expand_dims(temp, -1) - np.expand_dims(cent.T, 0), axis=1), axis=1)
    temp = cent[clust, :]
    
    
    #temp += np.matmul(((60-i)/60)**2, np.mean(C, axis=0, keepdims=True) / 2)
    temp += np.matmul(1/i, np.mean(C, axis=0, keepdims=True))
    temp = np.matmul(temp, c)
    #print(np.any(np.isnan(temp)))
    temp = temp/np.sum(temp, axis=1, keepdims=True)
    
    nD = D/np.sum(D,axis=1,keepdims=True)
    temp = np.sum(nD * np.log(np.nan_to_num(nD / temp)+eps), axis=1)
    return temp#, clust


#%%
Xpack = 'XultaPack.csv'
Epack = 'XultaDraftPack.csv'
Packs = [pd.read_csv(Xpack, index_col=None), pd.read_csv(Epack, index_col=None), pd.read_csv(Epack, index_col=None), pd.read_csv(Xpack, index_col=None)]
Packs = [j[[(i in k) for i in j['Id']]] for j in Packs]

def samplePack(P):
    pack = []
    temp = P['Rarity'].str.lower()=='rare'
    pack += list(np.random.choice(P['Id'][temp], 1, p=P['Odds'][temp]/np.sum(P['Odds'][temp])))
    temp = P['Rarity'].str.lower()=='uncommon'
    pack += list(np.random.choice(P['Id'][temp], 3, p=P['Odds'][temp]/np.sum(P['Odds'][temp]), replace=False))
    temp = P['Rarity'].str.lower()=='common'
    pack += list(np.random.choice(P['Id'][temp], 8, p=P['Odds'][temp]/np.sum(P['Odds'][temp]), replace=False))
    return pack


#%%
D = [np.zeros((1, c.shape[1])) for _ in range(12)]
test = []
for pack in Packs:
    P = [samplePack(pack) for _ in range(12)]
    
    for _ in range(12):
        for i in range(12):
            temp = [cardCode[j] for j in P[i]]
            
            D1 = []
            for j in temp:
                d = deepcopy(D[i])
                d[0, j] += 1
                D1.append(d)
            D1 = np.vstack(D1)
            
            kl = KLdiv(D1, C, N, tfidf, cent)#, ignore=sigil)
            #print(kl)
            n = np.argmin(kl)
            D[i][0, temp[n]] += 1
            P[i].pop(n)
        
        P = P[1:] + [P[0]]
    D = D[::-1]

inf = np.vstack([Influence[i] for i in k])
inf[inf>0] = 1

for i in range(12):
    for _ in range(18):
        D1 = []
        temp = np.where(D[i]>0)[1]
        for j in temp:
            d = deepcopy(D[i])
            d[0, j] -= 1
            D1.append(d)
        D1 = np.vstack(D1)
        
        kl = KLdiv(D1, C, N, tfidf, cent)
        
        n = np.argmin(kl)
        D[i][0, temp[n]] -= 1
    
    temp = np.matmul(D[i], inf)
    temp = np.round(temp / np.sum(temp) * 15)
    
    D[i][0, sigil[np.argsort(np.argmax(inf[sigil, :], axis=1))]] = temp





#%%
dlinks = []
for d in D:
    temp = zip(list(np.array(k)[np.where(d>0)[1]]), d[0, np.where(d>0)[1]].astype('int'))
    dlinks.append(head+';'.join([i+':'+str(n) for i,n in temp]))