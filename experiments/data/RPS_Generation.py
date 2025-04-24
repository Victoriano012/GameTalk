from pickle import dump
data = {}
with open('RPS.pkl', 'wb') as f:
    dump(data, f)