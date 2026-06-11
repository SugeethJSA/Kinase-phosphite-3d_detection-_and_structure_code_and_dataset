import time
import copy
import random
import numpy as np
import distance
import math
import pandas as pd
from scipy.spatial import distance
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
plt.style.use('seaborn-white')

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


import os
import time
import copy
import random
import numpy as np
import math

from sklearn.metrics import roc_curve, auc
from IPython.display import Audio

import torch
import torch.nn as nn
import torch.utils.data as data_utils

from multiclassifier import Model

seedy = 666
random.seed(seedy)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def allDone():
    return Audio('meow.wav', autoplay=True)
train_motifs = np.genfromtxt(r'data/train_motifs.csv', dtype='U')
train_motifxFamMatrix = np.genfromtxt(r'data/train_motifxFamMatrix.csv', delimiter=',', dtype=int)
test_motifs = np.genfromtxt(r'data/new_test_motifs.csv', dtype='U')
test_motifxFamMatrix = np.genfromtxt(r'data/new_test_motifxFamMatrix.csv', delimiter=',', dtype=int)
fams = np.genfromtxt(r'data/fams.csv', dtype='U')


## Split data into folds in a stratified k-fold manner.

def proba_mass_split(y, folds=5):
    obs, classes = y.shape
    dist = y.sum(axis=0).astype('float')
    dist /= dist.sum()
    index_list = []
    fold_dist = np.zeros((folds, classes), dtype='float')
    for _ in range(folds):
        index_list.append([])
    for i in range(obs):
        if i < folds:
            target_fold = i
        else:
            normed_folds = fold_dist.T / fold_dist.sum(axis=1)
            how_off = normed_folds.T - dist
            target_fold = np.argmin(np.dot((y[i] - .5).reshape(1, -1), how_off.T))
        fold_dist[target_fold] += y[i]
        index_list[target_fold].append(i)
    return index_list

np.random.seed(seedy)
folds = proba_mass_split(train_motifxFamMatrix)
#############################################
# Get Siamese embedding coords.
#############################################

embedding = np.genfromtxt(r"MODELS_siam/LEGACY/emb_latest_embedding.csv",delimiter=',',dtype=float)
train_embedding = embedding[ :len(train_motifs) ]
test_embedding = embedding[ len(train_motifs): ]
print(embedding.shape)

#############################################
# Get fam distance matrix for Phylo MSE loss.
#############################################

all_fams = (np.genfromtxt(r"data\fam_distances_blos62\fams.csv",dtype='U'))
dist_matrix = (np.genfromtxt(r"data\fam_distances_blos62\dist_matrix.csv",delimiter=',',dtype=float))

fam_idc = [np.where(all_fams==fam)[0][0] for fam in fams]
fam_dist_matrix = dist_matrix[fam_idc][:,fam_idc]
        
# normalize fam distances
fMax = np.max(fam_dist_matrix)
fMin = np.min(fam_dist_matrix)

fam_dist_matrix_scaled = np.array((fam_dist_matrix))
for i in range(len(fams)):
    for j in range(len(fams)):
        fam_dist_matrix_scaled[i][j] = 1 - float(fam_dist_matrix[i][j]-fMin)/(fMax-fMin) 
fam_dist_matrix = fam_dist_matrix_scaled

famDistMatrix = fam_dist_matrix
AMINOS = 'XWGSAELQDMPFTRIHVNCY_K'

def get_oneHot_motifs(motifs, AMINOS=AMINOS):
    oneHot_motifs = []
    for motif in motifs:
        one_hotted = np.zeros((len(motif), len(AMINOS)),dtype=float)
        for i,aa in enumerate(motif):
            hot = AMINOS.find(aa)
            one_hotted[i][hot] = 1
        oneHot_motifs.append(one_hotted)
    oneHot_motifs = np.asarray(oneHot_motifs)
    oneHot_motifs = np.swapaxes(oneHot_motifs,1,2)
    return oneHot_motifs

def get_stacked_features(motifs, embeddings):
    oneHot_motifs = get_oneHot_motifs(motifs)
    squished_oneHots = oneHot_motifs.reshape(oneHot_motifs.shape[0],oneHot_motifs.shape[1]*
                                         oneHot_motifs.shape[2])
    stacked_features = np.hstack((squished_oneHots,embeddings))
    stacked_features = torch.tensor(stacked_features)
    return stacked_features        


def get_loader(motifs,embedding,motifxFamMatrix,idc,my_batch):
    these_motifs = motifs[idc]
    this_embedding = embedding[idc]
    X = get_stacked_features(these_motifs,this_embedding)
    Y = torch.tensor(motifxFamMatrix[idc])
    dataset = data_utils.TensorDataset(X, Y)
    loader = data_utils.DataLoader(dataset, batch_size=my_batch, shuffle=True, drop_last=True)
    return loader
def get_microROC(y_test, y_score):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(len(fams)):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr_micro, tpr_micro, _ = roc_curve(y_test.ravel(), y_score.ravel())
    return auc(fpr_micro, tpr_micro)

def train_model(train_loader, val_loader, model, optimizer, batch_size, num_epochs, stopper='loss', 
                version='seq-coord', this_loss='phylo'):
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    running = - math.inf
    current = running
    best_run = 0
    best_model = copy.deepcopy(model.state_dict())
    
    for epoch in range(num_epochs):
        
        print("Epoch",epoch+1)
        for phase in ['train','validate']:
            running_loss = 0.0
            running_acc = 0.0
            if phase=='train':
                loader = train_loader
                model.train()
            else:
                loader = val_loader
                model.eval()
                
            for inputs,labels in loader:
                inputs = inputs.float().to(device)
                labels = labels.float().to(device)
                motif = inputs[:,:-embedding.shape[1]].reshape( batch_size, len(AMINOS), len(train_motifs[0]) )
                coords = inputs[:,-embedding.shape[1]:]

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model.forward(motif, coords, version)
                    if this_loss=='phylo':
                        loss = phylo_error(outputs,labels)
                    elif this_loss!='phylo':
                        criterion = nn.BCELoss() # BCE loss is the usual loss to use.
                        loss = criterion(outputs, labels)
                    if phase=='train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item()

                y_test = np.asarray(labels.cpu())
                y_score = outputs.cpu().detach().numpy()
                acc = get_microROC(y_test, y_score)
                running_acc += acc
            
            loss = running_loss / len(loader) 
            acc =  running_acc / len(loader) 
            
            if phase=='train':
                train_losses.append(loss)
                train_accs.append(acc)
                
            elif phase=='validate':
                val_losses.append(loss)
                val_accs.append(acc)
                
                if stopper=='loss':
                    current = -loss
                elif stopper=='acc':
                    current = acc
                elif stopper=='epoch':
                    current = epoch
                if current >= running:
                    running = current
                    best_run = epoch+1
                    best_model = copy.deepcopy(model.state_dict()) 
                
            print("~ %s LOSS: %5.3f | ACC: %5.3f" % (phase,loss,acc))
            if current >= running and phase=='validate':
                print("      BEST SO FAR ^ ^ ^")
        
    return (best_run, best_model, train_losses, train_accs, val_losses, val_accs)
def phylo_error(output, target):
        
    weights = np.ones((output.shape[0],output.shape[1]))

    for i,t in enumerate(target):
        t = t.cpu()
        wIdc = np.where(t.detach().numpy()==1)[0]

        if len(wIdc)==0:
            weights[i] = 0.000001
            continue
        theseWeights = np.ones((len(fams)))
        
        for wIdx in wIdc:
            thisWeight = famDistMatrix[wIdx].copy() # inter-fam
            thisWeight[wIdx] =  1.00 - famDistMatrix[wIdx][wIdx].copy() # intra-fam
            theseWeights+=thisWeight # add to existing list of fam distances, respectively (element wise)
            
        fWeight = theseWeights/len(fams) # take median / average
        weights[i] = fWeight 
        
    weights = torch.tensor(weights)
    weights = weights.to(device)
    crit = nn.BCELoss(reduction=my_reduction)
    answer = crit(output, target) * weights.mean().float()

    return answer
# # # # # # # # # # # # # # # # # # # # # # # # # # #
my_version = 'seq-coord'        
my_loss = 'phylo'            

my_stopper = 'loss'
my_batch = 32
my_epochs = 1
my_lr = 0.0015

my_reduction = 'sum'
# # # # # # # # # # # # # # # # # # # # # # # # # # #
s = time.time()

all_train_losses = []
all_train_accs = []
all_val_losses = []
all_val_accs = []

all_best_runs = []
all_models = []

for i,fold in enumerate(folds):
    
    print("\n* * * * * * * * FOLD %d * * * * * * * *\n" %(i+1))
    
    fold_val_idc = fold
    fold_train_idc = [x for x in range(len(train_motifs)) if x not in fold_val_idc]
    
    train_loader = get_loader(train_motifs,train_embedding,train_motifxFamMatrix,fold_train_idc,my_batch)
    val_loader = get_loader(train_motifs,train_embedding,train_motifxFamMatrix,fold_val_idc,my_batch)
    
    model = Model()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr = my_lr)
    
    (best_run, best_model, train_losses, train_accs, val_losses, val_accs) = \
                    train_model(train_loader, val_loader,
                                model, optimizer,my_batch, 
                                my_epochs,my_stopper,my_version,
                                my_loss)
    
    all_best_runs.append(best_run)
    all_models.append(best_model)
    all_train_losses.append(train_losses)
    all_train_accs.append(train_accs)
    all_val_losses.append(val_losses)
    all_val_accs.append(val_accs)
    
print("TIME: %5.3f mins" % ((time.time()-s)/60))
print("BEST RUNS:",all_best_runs)

allDone()
run = 'experiment_1'  # Example valid name for the directory

# Check if the directory exists, if not, create it
if not os.path.exists("MODELS_multiclass/%s/" % run):
    os.makedirs("MODELS_multiclass/%s/" % run)

# Iterate over all models and save their weights
for i, model_weights in enumerate(all_models):
    torch.save(model_weights, "MODELS_multiclass/%s/%d_weights" % (run, i))
all_train_losses_arr = np.zeros(my_epochs)
all_train_accs_arr = np.zeros(my_epochs)
all_val_losses_arr = np.zeros(my_epochs)
all_val_accs_arr = np.zeros(my_epochs)

for i in range(my_epochs):
    all_train_losses_arr[i] = sum([all_train_losses[j][i] for j in range(len(folds))]) / len(folds)
    all_train_accs_arr[i] = sum([all_train_accs[j][i] for j in range(len(folds))]) / len(folds)
    all_val_losses_arr[i] = sum([all_val_losses[j][i] for j in range(len(folds))]) / len(folds)
    all_val_accs_arr[i] = sum([all_val_accs[j][i] for j in range(len(folds))]) / len(folds)
    
if my_reduction=='sum':
    all_train_losses_arr=all_train_losses_arr*.005
    all_val_losses_arr=all_val_losses_arr*.005
    
import matplotlib.pyplot as plt
# %matplotlib inline

plt.figure(figsize=(7.5,5))

plt.plot(all_train_losses_arr,label='Train loss',c='blue')
plt.plot(all_val_losses_arr,label='Val loss',c='green')
plt.plot(all_train_accs_arr,label='Train acc',c='red')
plt.plot(all_val_accs_arr,label='Val acc',c='orange')

plt.xlabel("Epoch")
plt.legend(loc='center right')
plt.savefig("FIGS_multiclass/" + run + "_loss-acc")

plt.show()


import random
import numpy as np
import math

import torch
import torch.utils.data as data_utils

import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

from multiclassifier import Model

seedy = 666
random.seed(seedy)
train_motifs = np.genfromtxt(r'data/train_motifs.csv', dtype='U')
train_motifxFamMatrix = np.genfromtxt(r'data/train_motifxFamMatrix.csv', delimiter=',', dtype=int)
test_motifs = np.genfromtxt(r'data/new_test_motifs.csv', dtype='U')
test_motifxFamMatrix = np.genfromtxt(r'data/new_test_motifxFamMatrix.csv', delimiter=',', dtype=int)
fams = np.genfromtxt(r'data/fams.csv', dtype='U')

## Split data into folds in a stratified k-fold manner.

def proba_mass_split(y, folds=5):
    obs, classes = y.shape
    dist = y.sum(axis=0).astype('float')
    dist /= dist.sum()
    index_list = []
    fold_dist = np.zeros((folds, classes), dtype='float')
    for _ in range(folds):
        index_list.append([])
    for i in range(obs):
        if i < folds:
            target_fold = i
        else:
            normed_folds = fold_dist.T / fold_dist.sum(axis=1)
            how_off = normed_folds.T - dist
            target_fold = np.argmin(np.dot((y[i] - .5).reshape(1, -1), how_off.T))
        fold_dist[target_fold] += y[i]
        index_list[target_fold].append(i)
    return index_list

np.random.seed(seedy)
folds = proba_mass_split(train_motifxFamMatrix)
#############################################
# Get Siamese embedding coords.
#############################################

embedding = np.genfromtxt(r"MODELS_siam/LEGACY/emb_latest_embedding.csv",delimiter=',',dtype=float)
train_embedding = embedding[ :len(train_motifs) ]
test_embedding = embedding[ len(train_motifs): ]
print(embedding.shape)

#############################################
# Get fam distance matrix for Phylo MSE loss.
#############################################

all_fams = (np.genfromtxt(r"data\fam_distances_blos62\fams.csv",dtype='U'))
dist_matrix = (np.genfromtxt(r"data\fam_distances_blos62\dist_matrix.csv",delimiter=',',dtype=float))

fam_idc = [np.where(all_fams==fam)[0][0] for fam in fams]
fam_dist_matrix = dist_matrix[fam_idc][:,fam_idc]
        
# normalize fam distances
fMax = np.max(fam_dist_matrix)
fMin = np.min(fam_dist_matrix)

fam_dist_matrix_scaled = np.array((fam_dist_matrix))
for i in range(len(fams)):
    for j in range(len(fams)):
        fam_dist_matrix_scaled[i][j] = 1 - float(fam_dist_matrix[i][j]-fMin)/(fMax-fMin) 
fam_dist_matrix = fam_dist_matrix_scaled

famDistMatrix = fam_dist_matrix
AMINOS = 'XWGSAELQDMPFTRIHVNCY_K'

def get_oneHot_motifs(motifs, AMINOS=AMINOS):
    oneHot_motifs = []
    for motif in motifs:
        one_hotted = np.zeros((len(motif), len(AMINOS)),dtype=float)
        for i,aa in enumerate(motif):
            hot = AMINOS.find(aa)
            one_hotted[i][hot] = 1
        oneHot_motifs.append(one_hotted)
    oneHot_motifs = np.asarray(oneHot_motifs)
    oneHot_motifs = np.swapaxes(oneHot_motifs,1,2)
    return oneHot_motifs

def get_stacked_features(motifs, embeddings):
    oneHot_motifs = get_oneHot_motifs(motifs)
    squished_oneHots = oneHot_motifs.reshape(oneHot_motifs.shape[0],oneHot_motifs.shape[1]*
                                         oneHot_motifs.shape[2])
    stacked_features = np.hstack((squished_oneHots,embeddings))
    stacked_features = torch.tensor(stacked_features)
    return stacked_features        

def get_loader(motifs,embedding,motifxFamMatrix,idc,my_batch):
    these_motifs = motifs[idc]
    this_embedding = embedding[idc]
    X = get_stacked_features(these_motifs,this_embedding)
    Y = torch.tensor(motifxFamMatrix[idc])
    dataset = data_utils.TensorDataset(X, Y)
    loader = data_utils.DataLoader(dataset, batch_size=my_batch, shuffle=True, drop_last=True)
    return loader
def eval_model(model, thresh=0.5):
    
    model.eval()
    loader = get_loader(test_motifs,test_embedding,test_motifxFamMatrix,range(len(test_motifs)),len(test_motifs))
    
    for inputs, labels in loader:
        
        inputs = inputs.float().to(device)
        labels = labels.float().to(device)

        motif = inputs[:,:-embedding.shape[1]].reshape( len(test_motifs), len(AMINOS), len(train_motifs[0]) )
        coords = inputs[:,-embedding.shape[1]:]
        
        outputs = model.forward(motif, coords)
        
        accuracy = 0
        totTrues = 0
        for i,out in enumerate(outputs):
            pred = np.where(out.cpu().detach().numpy() > thresh)[0]
            true = np.where(labels.data.cpu()[i]==1)[0]
            totTrues += len(true)
            accuracy += len(pred)

        y_score = outputs.cpu().detach().numpy()
        y_test = np.asarray(labels.cpu())
                
        return y_test, y_score
import torch
import numpy as np

# Ensure the device is set correctly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set the run variable
run = '00'

# Initialize lists to store the scores and test labels
all_y_scores = []
all_y_test = []

# Loop through the models and evaluate them
for i in range(5):
    # Initialize the model and move it to the appropriate device
    model = Model()
    model = model.to(device)
    
    # Load the saved model weights explicitly to the CPU if necessary
    model.load_state_dict(torch.load(f"MODELS_multiclass/experiment_1/{i}_weights", map_location=device))
    
    # Evaluate the model and extend the results into the lists
    y_test, y_score = eval_model(model)
    all_y_scores.extend(y_score)
    all_y_test.extend(y_test)

# Convert the results to numpy arrays
y_score = np.asarray(all_y_scores)
y_test = np.asarray(all_y_test)

## ROC



FAM_IDC = [x for x in range(len(fams))]

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in FAM_IDC:
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in FAM_IDC]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in FAM_IDC:
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= len(FAM_IDC)

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
## PRECISION-RECALL



precision = dict()
recall = dict()
average_precision = dict()

for i in range(len(fams)):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
    average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_score.ravel())

average_precision["micro"] = average_precision_score(y_test, y_score, average="micro")
average_precision["macro"] = average_precision_score(y_test, y_score, average="macro")
import matplotlib.pyplot as plt
cmap = plt.get_cmap("tab20")
fam_to_col = {fam: cmap(idx % 20) for idx, fam in enumerate(fams)}
# from itertools import cycle

plt.style.use('default')
# fig, (ax1, ax2) = plt.subplots(2,figsize=(5,11))
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(11,5))

lw = 1.25

############################################################
###################  AUROC #################################
############################################################

lines = []
labels = []

l, = ax1.plot(0,0,color='white')
lines.append(l)
labels.append('micro-average (area = {0:0.3f})'.format(roc_auc["micro"]))
l, = ax1.plot(0,0,color='white')
lines.append(l)
labels.append('macro-average (area = {0:0.3f})'.format(roc_auc["macro"]))

for i,fam in enumerate(fams):
    fclass = fams[i]
    if fclass=='AKT':
        fclass = 'Akt'
    if fclass=='SRC':
        fclass = 'Src'    
    l, = ax1.plot(fpr[i], tpr[i], color=fam_to_col[fam], lw=lw,
             label='{0} (area = {1:0.3f})'.format(fclass, roc_auc[i]))
    labels.append('{0} (area = {1:0.3f})'.format(fclass, roc_auc[i]))
    lines.append(l)

fig.subplots_adjust(hspace=.275)
ax1.plot([0, 1], [0, 1], 'k--', lw=lw)
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.tick_params(axis="x", labelsize=10)
ax1.tick_params(axis="y", labelsize=10)
ax1.set_xlabel('False Positive Rate',fontsize=12)
ax1.set_ylabel('True Positive Rate',fontsize=12) 
ax1.set_title('ROC curve per kinase family',fontsize=12) 
labels, lines = zip(*sorted(zip(labels, lines), key=lambda t: t[0], reverse=False))
ax1.legend(lines, labels, loc=(0.475, .0175), fontsize=7.5)  # (1.05, .0)

############################################################
###############  PRECISION-RECALL ##########################
############################################################

lines = []
labels = []

l, = ax2.plot(0,0,color='white')
lines.append(l)
labels.append('micro-average (area = {0:0.3f})'.format(average_precision["micro"]))
l, = ax2.plot(0,0,color='white')
lines.append(l)
labels.append('macro-average (area = {0:0.3f})'.format(average_precision["macro"]))

for i,fam in enumerate(fams):
    l, = ax2.plot(recall[i], precision[i], color=fam_to_col[fam], lw=lw)
    lines.append(l)
    fclass = fams[i]
    if fclass=='AKT':
        fclass = 'Akt'
    if fclass=='SRC':
        fclass = 'Src'
    labels.append('{0} (area = {1:0.3f})'.format(fclass, average_precision[i]))

ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.tick_params(axis="x", labelsize=10)
ax2.tick_params(axis="y", labelsize=10)
ax2.set_xlabel('Recall',fontsize=12) 
ax2.set_ylabel('Precision',fontsize=12) 
ax2.set_title('Precision-recall curve per kinase family',fontsize=12) 

labels, lines = zip(*sorted(zip(labels, lines), key=lambda t: t[0], reverse=False))
ax2.legend(lines, labels, loc=(0.025, .0175), fontsize=7.5) # (1.05, .0)

plt.savefig("FIGS_multiclass/%s_roc-prc" % run, bbox_inches='tight')
plt.show()
confusion = np.zeros((len(fams), len(fams)),dtype=float)

for i,score in enumerate(y_score):
    pred = np.where(score > 0.5000)[0]
    if len(pred)<1:
        pred = [np.argmax(score)]
    true = np.where(y_test[i]==1)[0]
    for p_idx in pred:
        for t_idx in true:
            confusion[p_idx][t_idx] += 1
            
labs = confusion.copy()

for i in range(len(fams)):
    confusion[i] = confusion[i] / confusion[i].sum()
for i in range(len(fams)):
    for j in range(len(fams)):
        if math.isnan(confusion[i][j]) or confusion[i][j]==0:
            confusion[i][j]=0
heat_map = sb.heatmap(confusion, annot=labs, cmap="GnBu_r", fmt='1.0f',
                      square=False, yticklabels=fams, xticklabels=fams )

b, t = plt.ylim() 
plt.ylim(b, t) 

plt.xlabel("True label",fontsize=12, labelpad=10)
plt.ylabel("Predicted label",fontsize=12, labelpad=10)
plt.title("Confusion matrix for kinase labels",fontsize=12, pad=12)
plt.savefig("FIGS_multiclass/%s_confusion" % run, bbox_inches='tight')

plt.show()


