import json

def insert_markdowns(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    cells = nb.get('cells', [])
    new_cells = []
    
    def create_md_cell(src):
        if not isinstance(src, list):
            src = [src]
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": src
        }

    for c in cells:
        if c.get('cell_type') == 'code' and c.get('source'):
            first_line = c['source'][0].strip()
            
            if first_line.startswith('import time'):
                new_cells.append(create_md_cell([
                    "## Stage 1: Siamese Neural Network Embedder\n",
                    "In this section, we build and train a **Siamese Neural Network**. Its purpose is to learn a 100-dimensional mathematical representation (embedding) for kinase sequence motifs such that motifs belonging to the same kinase family are pushed closer together, while motifs from different families are pushed apart."
                ]))
            elif first_line.startswith('def get_gram_seq'):
                new_cells.append(create_md_cell([
                    "### Helper Functions for Data Processing\n",
                    "These functions are responsible for formatting the biological sequence motifs (like converting amino acids to one-hot or n-grams) and generating matched/mismatched batches for the Siamese network."
                ]))
            elif first_line.startswith('class SiameseLoss'):
                new_cells.append(create_md_cell([
                    "### Model Definitions: Siamese Architecture & Loss\n",
                    "Here we define the custom distance loss function (Contrastive Loss) and the `Model` architecture itself, which uses PyTorch to process the sequence motifs."
                ]))
            elif first_line.startswith('def get_embedding'):
                new_cells.append(create_md_cell([
                    "### Extracting the Embeddings\n",
                    "After the Siamese model is trained, this block converts all sequences into their dense 100-dimensional embeddings and saves them to a CSV file to be used by the multiclassifier later."
                ]))
            elif first_line.startswith('import umap.umap_'):
                new_cells.append(create_md_cell([
                    "### Visualization: UMAP Reduction\n",
                    "To visually verify that the model has successfully separated the kinase families, we use UMAP to squash the 100-dimensional embeddings down to 2 dimensions so we can plot them on a scatter plot."
                ]))
            elif first_line.startswith('import os'):
                new_cells.append(create_md_cell([
                    "## Stage 2: Multiclassifier Training\n",
                    "Now that we have rich 100D embeddings for our motifs, we can train a final dense neural network (the Multiclassifier) to predict the exact kinase family. This section sets up the data loaders and 5-Fold Cross Validation logic."
                ]))
            elif first_line.startswith('def train_model'):
                new_cells.append(create_md_cell([
                    "### Multiclassifier Training Loop\n",
                    "This function handles the actual PyTorch training loop for the dense multiclassifier, including computing BCE loss, accuracy, and handling backpropagation."
                ]))
            elif first_line.startswith('plt.figure(figsize='):
                new_cells.append(create_md_cell([
                    "### Execution & ROC Plotting\n",
                    "This cell executes the K-fold cross validation pipeline, tests the model, and generates an ROC curve mapping True Positive vs False Positive rates."
                ]))
            elif first_line.startswith('import random'):
                new_cells.append(create_md_cell([
                    "## Stage 3: Evaluation\n",
                    "This final block takes the trained model and tests it on completely unseen `test_motifs`. It generates confusion matrices and final metric scores so you can see exactly where the model succeeded and where it got confused."
                ]))

        new_cells.append(c)
            
    nb['cells'] = new_cells
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Markdown cells inserted successfully!")

insert_markdowns('finalised_application.ipynb')
