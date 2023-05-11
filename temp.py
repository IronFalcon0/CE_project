import json
import itertools

with open('sets.txt', 'r') as f:
    sets = json.loads(f.read())

print(set(itertools.chain(*sets)))
print(len(set(itertools.chain(*sets))))