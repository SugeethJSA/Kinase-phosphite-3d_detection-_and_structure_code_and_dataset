import time
import copy
import random
import numpy as np
import distance
import math
import pandas as pd
from collections import Counter

from sklearn.model_selection import train_test_split

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import optim

from IPython.display import Audio

import numpy as np

train_motifs = np.genfromtxt(r'data/train_motifs.csv', dtype='U')
train_motifxFamMatrix = np.genfromtxt(r'data/train_motifxFamMatrix.csv', delimiter=',', dtype=int)
test_motifs = np.genfromtxt(r'data/new_test_motifs.csv', dtype='U')
test_motifxFamMatrix = np.genfromtxt(r'data/new_test_motifxFamMatrix.csv', delimiter=',', dtype=int)
fams = np.genfromtxt(r'data/fams.csv', dtype='U')


all_motifs = np.hstack([train_motifs,test_motifs])
all_motifxFamMatrix = np.vstack([train_motifxFamMatrix,test_motifxFamMatrix])

X_train, X_val = train_test_split(range(len(train_motifs)), test_size=0.1, random_state=666)

print(len(X_train), len(X_val), len(test_motifs))
def get_gram_seq(motif,gram_length=1):
    gram_seq = []
    for i in range(len(motif)):
        gram = motif[i:i+gram_length]
        gram_seq.append(gram)
    return gram_seq

def get_encoded_motifs(grammed_motifs, all_grams):
    gram_counts = Counter(all_grams)
    gram_list = sorted(gram_counts, key=gram_counts.get, reverse=True)
    gram_to_int = {gram:idx+1 for idx, gram in enumerate(gram_list)}  
    encoded_motifs = [[gram_to_int[gram] for gram in motif] for motif in grammed_motifs]
    return encoded_motifs, gram_to_int, gram_list

AMINOS = 'XWGSAELQDMPFTRIHVNCY_K'
gram_length = 1  # Define the gram length

all_grams = []
train_grammed_motifs = []
for motif in train_motifs:
    grammed_motif = get_gram_seq(motif,gram_length)
    train_grammed_motifs.append(grammed_motif)
    all_grams.extend(grammed_motif)
    
test_grammed_motifs = []
for motif in test_motifs:
    grammed_motif = get_gram_seq(motif,gram_length)
    test_grammed_motifs.append(grammed_motif)
    all_grams.extend(grammed_motif)
    
vocab_size = len(set(all_grams))
print("vocab size:",vocab_size)

train_encoded_motifs, gram_to_int, gram_list = get_encoded_motifs(train_grammed_motifs, all_grams)
test_encoded_motifs, gram_to_int, gram_list = get_encoded_motifs(test_grammed_motifs, all_grams)
all_encoded_motifs = train_encoded_motifs + test_encoded_motifs
def get_random_motif_and_fam(idc):
    mIdx = random.choice(idc)  
    motif = train_encoded_motifs[mIdx] 
    fIdx = np.where(train_motifxFamMatrix[mIdx]==1)
    theseFams = fams[fIdx]
    return (mIdx,motif,fIdx,theseFams)

def get_batch(idc,batch_size):
    batch = []
    switch = 0
    
    while switch < batch_size:
        mIdx_1, motif_1, fIdx_1, fams_1 = get_random_motif_and_fam(idc)
        mIdx_2, motif_2, fIdx_2, fams_2 = get_random_motif_and_fam(idc)
        
        if len(fams_1)==0 and len(fams_2)==0: 
            continue
        
        label = distance.jaccard(set(fams_1),set(fams_2))
        if switch%2 != round(label):     # math.ceil(label):
            continue
        switch += 1
            
        triplet = [motif_1, motif_2, label]
        batch.append(triplet)

    return batch


def evaluate(model, idc, criterion, iters, batch_size, margin, dist_type):
    model.eval()
    loss_history = 0

    with torch.no_grad():
        
        for i in range(iters):
            batch = get_batch(idc, batch_size)
            motifs_net1 = torch.stack([torch.tensor(x[0]).to(device) for x in batch])
            motifs_net2 = torch.stack([torch.tensor(x[1]).to(device) for x in batch])
            labels = torch.stack([torch.tensor(x[2]).to(device) for x in batch])
            
            embeds_1, embeds_2 = model(motifs_net1, motifs_net2)
            loss = criterion(embeds_1, embeds_2, labels,margin, dist_type)
                
            loss_history += loss.item()
            
    return loss_history / iters

    
def train(model, idc, optimizer, criterion, iters, batch_size, margin, dist_type):
    model.train()
    loss_history = 0
    
    for i in range(iters):
            
        batch = get_batch(idc, batch_size)
        motifs_net1 = torch.stack([torch.tensor(x[0]).to(device) for x in batch])
        motifs_net2 = torch.stack([torch.tensor(x[1]).to(device) for x in batch])
        labels = torch.stack([torch.tensor(x[2]).to(device) for x in batch])
        
        optimizer.zero_grad()
        
        embeds_1, embeds_2 = model(motifs_net1, motifs_net2)
        loss = criterion(embeds_1, embeds_2, labels, margin, dist_type)
        
        loss.backward()
        optimizer.step()
                
        loss_history += loss.item()
        
    return loss_history / iters

class SiameseLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, z1, z2, label, margin=2.0, dist_type='L2'):
        ''' Calculates the pairwise loss given two embeddings and their 0/1 label.
            margin: somewhere between 0.0 and 3.0
            dist_type: manhattan (L1) or euclidean (L2)
        '''
        if dist_type=='L2':
            distance = F.pairwise_distance(z1, z2)
        elif dist_type=='L1':
            distance = torch.sum( torch.abs(z1-z2), axis=1)
        siam_loss = torch.mean((1-label) * torch.pow(distance, 2) +
                                (label) * torch.pow(torch.clamp(margin - distance, min=0.0), 2))
        return siam_loss
class Model(nn.Module):
            
    def __init__(self):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size+1, embedding_dim=64)   
        
        self.lstm = nn.LSTM(64, 128, bidirectional=True,num_layers=2,
                            batch_first=True,dropout=0.2)
        
        self.fc1 = nn.Linear(128*2*15, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 100)
        
        self.relu = nn.ReLU()
        self.drpt = nn.Dropout(p=0.0)
                
    def forward_once(self, motif): 
        
        embedded = self.embedding(motif)
        lstmed, _ = self.lstm(embedded)
        flattened = lstmed.reshape(lstmed.shape[0], lstmed.shape[1]*lstmed.shape[2])        
    
        fc1 = self.fc1(flattened)
        fc1 = self.relu(fc1)
        fc1 = self.drpt(fc1)
        
        fc2 = self.fc2(fc1)
        fc2 = self.relu(fc2)
        fc2 = self.drpt(fc2)
        
        fc3 = self.fc3(fc2)
        fc3 = self.relu(fc3)

        out = self.out(fc3)

        return out
        
    def forward(self, motifs_net1, motifs_net2):
        embed_1 = self.forward_once(motifs_net1)
        embed_2 = self.forward_once(motifs_net2)
        return (embed_1, embed_2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model() 
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr = 0.005 ) 
criterion = SiameseLoss()

bs = 64

# A sub_iter is a single run through the network
sub_iters = 100
# Loss is calculated per super_iter (collection of sub_iters)
super_iters = 10

my_margin = 1.5
my_dist_type = 'L1'
print("Device:",device,"\n")

train_loss_history = []
val_loss_history = []
best_val_loss = float('inf')
total_time = time.time()

best_super = 0
for i in range(super_iters):
    start = time.time()
    train_loss = train(model, X_train, optimizer, criterion, 
                       iters=sub_iters, batch_size=bs, 
                       margin=my_margin, dist_type=my_dist_type)
    val_loss = evaluate(model, X_val, criterion, 
                        iters=sub_iters, batch_size=bs,
                        margin=my_margin, dist_type=my_dist_type)
    
    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    
    if val_loss <= best_val_loss:

        best_val_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        best_super = i

    print("Time: %5.3f secs, Super iter %d\n* Train loss %5.4f | Val loss: %5.4f" % 
          ( time.time()-start, i+1, train_loss, val_loss ))

final_model = Model().to(device)
final_model.load_state_dict(best_model_wts)
import os
import time
import torch 

# Ensure 'best_super' and 'total_time' are defined somewhere in the script
# Example values (replace these with actual values from your training loop)
best_super = 5  # Example: Last super iteration
total_time = time.time() - 1000  # Example: Replace with actual start time

# Print training details
print("--------------------\nLast super iter of learning:", best_super)
print("Total train time: %5.3f mins" % ((time.time() - total_time) / 60))



# Ensure the directory exists before saving the model
save_dir = "MODELS_siam/LEGACY"
os.makedirs(save_dir, exist_ok=True)

# Define a valid filename for the model
run = "latest"  # Change this to a timestamp, epoch number, etc., if needed
model_path = os.path.join(save_dir, f"siameseWeights_{run}.pth")

# Save model
torch.save(model.state_dict(), model_path)

print(f"Model saved to {model_path}")
def get_embedding(model, enc_mots):
    model.eval()
    to_embed = torch.tensor(np.array((enc_mots))).to(device)
    embedding = model.forward_once(to_embed)
    return embedding.cpu().detach().numpy()

embedded = get_embedding(model,all_encoded_motifs)   #range(len(motifs)))
df = pd.DataFrame(embedded,dtype=float)
df.to_csv("MODELS_siam/LEGACY/emb_%s_embedding.csv" % (run),header=None,index=None)
# !pip install umap-learn
import umap.umap_ as umap
umapper = umap.UMAP(
    n_neighbors=20, # changed from 200
    min_dist=0.5, # changed from 0.1
    n_components=2,
    metric='euclidean' )

s = time.time()
pos_umap = umapper.fit_transform(embedded)
# allDone()

print ("secs: %5.3f" % (time.time()-s))
import os

# Make sure 'run' is defined
run = "latest"  # You can replace this with any desired name or string
f = os.path.join('FIGS_siam', 'LEGACY', run)  # Use os.path.join for platform-independent path construction

# Create the directory if it doesn't exist
os.makedirs(f, exist_ok=True)

label_size = 45
title_size = 50
tick_size = 40

import matplotlib.pyplot as plt
# %matplotlib inline
plt.style.use('seaborn-v0_8-white')

colors = ['red','#CD0000','deepskyblue','blue','green','blueviolet','orange','magenta','blueviolet','violet','deeppink','crimson','mediumslateblue','brown']

plt.figure(figsize=(20, 20))
plt.scatter(pos_umap[:, 0], pos_umap[:, 1], marker='o', s=5, color='black', alpha=1)  # alpha=0.25
plt.title("Siamese latent space, UMAP reduction", fontsize=title_size, y=1.01)
plt.xlabel("UMAP-1", fontsize=label_size)
plt.ylabel("UMAP-2", fontsize=label_size)
plt.xticks(fontsize=tick_size)
plt.yticks(fontsize=tick_size)

plt.savefig(f + "/noHighlights")
plt.show()

pop_fams = ['PKC', 'AKT', 'CDK', 'MAPK', 'SRC', 'CK2', 'PKA', 'PIKK']
# pop_fams = ['CAMK-UNIQUE', 'DYRK', 'CAMKL', 'STE20', 'PKC', 'AKT', 'CDK', 'MAPK', 'SRC', 'CK2', 'PKA', 'PIKK']

i = -1
for _, fam in enumerate(pop_fams):
    i += 1
    fIdx = np.where(fams == fam)[0][0]
    plt.figure(figsize=(20, 20))
    plt.title(f"Siamese latent space, UMAP reduction: {fam}", fontsize=title_size, y=1.01)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.xlabel("UMAP-1", fontsize=label_size)
    plt.ylabel("UMAP-2", fontsize=label_size)
    
    plt.scatter(pos_umap[:, 0], pos_umap[:, 1], marker='o', s=25, color='grey', alpha=0.30)
    
    for mIdx, (x, y) in enumerate(zip(pos_umap[:, 0], pos_umap[:, 1])):
        if all_motifs[mIdx] not in test_motifs:
            continue
        elif all_motifxFamMatrix[mIdx][fIdx] == 1:
            plt.scatter(x, y, marker='o', s=200, c=colors[i], alpha=1.0, edgecolors='black')
    
    plt.savefig(f + f"/{fam}")
    plt.show()
import os
import shutil

# Define the directory
run = "latest"
output_dir = os.path.join('FIGS_siam', 'LEGACY', run)

# Create a ZIP file
zip_file = output_dir + ".zip"
shutil.make_archive(output_dir, 'zip', output_dir)

# Confirm the zip file is created
print(f"Download your images: {zip_file}")

