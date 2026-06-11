import json

def split_notebook():
    with open('finalised_application.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    cells = nb.get('cells', [])
    if not cells: return
    
    # We assume the first big code cell is the one to split
    big_cell = None
    big_cell_idx = -1
    for i, c in enumerate(cells):
        if c.get('cell_type') == 'code' and len(c.get('source', [])) > 50:
            big_cell = c
            big_cell_idx = i
            break
            
    if not big_cell:
        print("Could not find a large code cell to split.")
        return
        
    source_lines = big_cell['source']
    
    # Define split markers (substrings indicating where to start a new cell)
    split_markers = [
        "import torch",                                      # Cell 1: Setup & Data loading
        "def calc_seq_identity(seq1, seq2):",                # Cell 2: Distance functions
        "class SiameseLoss(torch.nn.Module):",               # Cell 3: Siamese Model
        "def get_batch(idc, batch_size):",                   # Cell 4: Siamese Helpers
        "def train(model, train_idc, optimizer, criterion,", # Cell 5: Siamese Train Func
        "if __name__ == '__main__':",                        # Cell 6: Run Siamese
        "import umap.umap_ as umap",                         # Cell 7: UMAP setup
        "plt.style.use('seaborn-v0_8-white')",               # Cell 8: Plotting 
        "from multiclassifier import Model",                 # Cell 9: Multiclassifier setup
        "def train_model(train_motifs, train_labels,",       # Cell 10: Multiclassifier Train loop
        "if __name__=='__main__':"                           # Cell 11: Run Multiclassifier (there might be multiple if __main__)
    ]
    
    # Actually, we can split by looking at clear logical breaks or just manually define line indices
    # Let's split by searching for these keywords and creating new cells
    
    new_cells = []
    current_source = []
    
    # Custom splits
    def create_code_cell(src):
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": src
        }

    for line in source_lines:
        # Check if line matches any split marker
        should_split = False
        for marker in split_markers:
            # We want to match the exact start of the line or just the substring?
            # Many markers are exactly at the start
            if line.startswith(marker):
                should_split = True
                break
                
        # If it's the second `if __name__ == '__main__':`, we need to make sure we don't split infinitely.
        # Let's just remove the matched marker so it doesn't split again if duplicated.
        if should_split and current_source:
            new_cells.append(create_code_cell(current_source))
            current_source = [line]
            # Remove the marker from split_markers to avoid splitting again on the same keyword
            for i, marker in enumerate(split_markers):
                if line.startswith(marker):
                    split_markers.pop(i)
                    break
        else:
            current_source.append(line)
            
    if current_source:
        new_cells.append(create_code_cell(current_source))
        
    # Replace the big cell with the new split cells
    nb['cells'] = cells[:big_cell_idx] + new_cells + cells[big_cell_idx+1:]
    
    with open('finalised_application.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
    print(f"Split the big cell into {len(new_cells)} smaller cells!")

split_notebook()
