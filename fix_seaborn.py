import json
with open('finalised_application.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        new_source = []
        for line in cell['source']:
            if "plt.style.use('seaborn-white')" in line:
                new_source.append(line.replace("plt.style.use('seaborn-white')", "plt.style.use('seaborn-v0_8-white')"))
            else:
                new_source.append(line)
        cell['source'] = new_source
with open('finalised_application.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
