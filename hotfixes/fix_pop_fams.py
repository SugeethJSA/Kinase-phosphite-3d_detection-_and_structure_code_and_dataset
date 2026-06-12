import json
with open('finalised_application.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        new_source = []
        for line in cell['source']:
            if "pop_fams = ['PKC', 'AKT', 'CDK', 'MAPK', 'SRC', 'CK2', 'PKA', 'PIKK']" in line:
                new_source.append(line.replace("pop_fams = ['PKC', 'AKT', 'CDK', 'MAPK', 'SRC', 'CK2', 'PKA', 'PIKK']", "pop_fams = ['PRKCA', 'AKT1', 'CDK1', 'MAPK1', 'SRC', 'ATM']"))
            else:
                new_source.append(line)
        cell['source'] = new_source
with open('finalised_application.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
