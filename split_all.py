import json

def split_notebook(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    cells = nb.get('cells', [])
    new_cells = []
    
    # Custom splits
    def create_code_cell(src):
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": src
        }

    changed = False
    for c in cells:
        if c.get('cell_type') == 'code' and len(c.get('source', [])) > 50:
            changed = True
            source_lines = c['source']
            current_source = []
            
            # Simple heuristic: split on "def " and "class " and "if __name__"
            # But ensure we don't make tiny cells
            for line in source_lines:
                if (line.startswith("def ") or line.startswith("class ") or line.startswith("if __name__")) and len(current_source) > 10:
                    new_cells.append(create_code_cell(current_source))
                    current_source = [line]
                else:
                    current_source.append(line)
            
            if current_source:
                new_cells.append(create_code_cell(current_source))
        else:
            new_cells.append(c)
            
    if changed:
        nb['cells'] = new_cells
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Successfully split big cells in {filename}")

for nb in ['01-siamese_embedder-Copy1.ipynb', '02-train_multiclassifier-Copy1.ipynb', '03-evaluate_multiclassifier-Copy1.ipynb']:
    split_notebook(nb)
