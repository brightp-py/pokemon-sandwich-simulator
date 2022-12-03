import numpy as np
import json

with open("./src/data/flavors.json", 'r', encoding='utf-8') as file:
    FLAVORS = json.load(file)

POWERS = []
with open("./src/data/powers.json", 'r', encoding='utf-8') as file:
    for power in json.load(file):
        if power == 'Item Drop Power':
            POWERS.append('Item Drop')
        else:
            POWERS.append(power.split()[0])

with open("./src/data/types.json", 'r', encoding='utf-8') as file:
    TYPES = json.load(file)

with open("./src/data/flavor_table.json", 'r', encoding='utf-8') as file:
    FLAVOR_BONUS = json.load(file)

with open("./src/data/fillings.json", 'r', encoding='utf-8') as file:
    fillings = json.load(file)

with open("./src/data/condiments.json", 'r', encoding='utf-8') as file:
    condiments = json.load(file)

ind2row = FLAVORS + POWERS + TYPES + ['Filling', 'Condiment'] * 2
row2ind = {row: ind for ind, row in enumerate(ind2row)}

ind2col = []
col2ind = {}

npieces = []
columns = []

is_condiment = False
for ingred in (fillings, condiments):
    for ind, food in enumerate(ingred):
        # ind2col.append(food['name'])
        # col2ind[food['name']] = ind

        if is_condiment:
            pieces = 1
        else:
            pieces = food['pieces']
        npieces.append(pieces)

        new_col = np.zeros_like(ind2row, dtype=int)
        for taste in food['tastes']:
            # new_col[row2ind[taste['flavor']]] = taste['amount'] * pieces
            new_col[row2ind[taste['flavor']]] = taste['amount']
        for power in food['powers']:
            # new_col[row2ind[power['type']]] = power['amount'] * pieces
            new_col[row2ind[power['type']]] = power['amount']
        for element in food['types']:
            # new_col[row2ind[element['type']]] = element['amount'] * pieces
            new_col[row2ind[element['type']]] = element['amount']
        
        # if is_condiment:
        #     new_col[-2:] = [0, 1]
        # else:
        #     new_col[-2:] = [1, 0]
        
        # columns.append(new_col)
        for p in range(pieces):
            name = food['name']
            if p > 0:
                name += f" (Drop {str(p)})"
            sub_col = new_col * (pieces - p)
            if is_condiment:
                sub_col[-4:] = [0, 1, 0, -1]
            else:
                sub_col[-4:] = [1, 0, -1, 0]
            
            columns.append(sub_col)
            ind2col.append(name)
            col2ind[name] = ind
            break

    is_condiment = True

NPIECES = np.array(npieces)
MATRIX = np.transpose(np.array(columns))
