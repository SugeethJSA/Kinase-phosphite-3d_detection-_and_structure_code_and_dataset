import json

def split_all_cells(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    cells = nb.get('cells', [])
    new_cells = []
    
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
        if c.get('cell_type') == 'code' and len(c.get('source', [])) > 40:
            changed = True
            source_lines = c['source']
            current_source = []
            
            # Split on functions, classes, imports, or __main__
            for line in source_lines:
                # We want to split cleanly. Let's split if we hit 'def ', 'class ', 'if __name__', or 'plt.style.use'
                if (line.startswith("def ") or line.startswith("class ") or line.startswith("if __name__") or line.startswith("plt.figure")) and len(current_source) > 10:
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
        print(f"Successfully split all big cells in {filename}")

split_all_cells('finalised_application.ipynb')
