import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from pathlib import Path

# Load the dataset
notebook_dir = Path().resolve()
movilens = pd.read_csv( notebook_dir / 'movilens_dataset/movies.csv')

# Transform the dataset
movilens['genres'] = movilens['genres'].str.split('|')
te = TransactionEncoder()
te_ary = te.fit(movilens['genres']).transform(movilens['genres'])
movilens = pd.DataFrame(te_ary, columns=te.columns_).set_index(movilens['title'])

print(f"{movilens.head()}\n")

# Support
print(" ## Support ##")
print(f"{movilens.mean()}\n")

# -------------------------------------

