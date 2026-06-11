import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_classes=25):
        super(Model, self).__init__()
        
        # Flattened motif size: 22 amino acids * 15 sequence length = 330
        # Siamese Embeddings (coords) size: 100
        # Combined size: 430
        
        self.fc1 = nn.Linear(330 + 100, 512)
        self.fc2 = nn.Linear(512, 128)
        self.out = nn.Linear(128, num_classes)
        
        # In case they want to run without Siamese Embeddings (sequence only)
        self.fc1_seq = nn.Linear(330, 512)
        
        # In case they want to run without sequence features (coordinates only)
        self.fc1_coord = nn.Linear(100, 512)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, motif, coords, version='seq-coord'):
        # motif shape: (batch_size, 22, 15)
        # flatten motif
        batch_size = motif.size(0)
        motif_flat = motif.view(batch_size, -1) # Shape: (batch_size, 330)
        
        if version == 'seq-coord':
            # Combine sequence and 3D Siamese coordinate embeddings
            combined = torch.cat((motif_flat, coords), dim=1)
            x = self.relu(self.fc1(combined))
        elif version == 'seq-only':
            x = self.relu(self.fc1_seq(motif_flat))
        elif version == 'coord-only':
            x = self.relu(self.fc1_coord(coords))
        else:
            raise ValueError(f"Unknown version: {version}")
            
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.out(x)
        
        # Output probabilities for Binary Cross Entropy Loss
        return self.sigmoid(x)
