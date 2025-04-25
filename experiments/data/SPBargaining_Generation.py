from pickle import load, dump
from pprint import pprint

with open('BertrandCompetition.pkl', 'rb') as f:
    products = load(f)

products.pop('demand_den')
products['value'] = products.pop('max_price_with_demand')
products['cost'] = [c//2 for c in products['cost']]

pprint(products)

with open('SPBargaining.pkl', 'wb') as f:
    dump(products, f)