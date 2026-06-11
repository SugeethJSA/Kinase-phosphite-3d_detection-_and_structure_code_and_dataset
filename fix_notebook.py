import json
with open('finalised_application.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        cell['source'] = [line for line in cell['source'] if 'from scipy.spatial import distance' not in line]
with open('finalised_application.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
